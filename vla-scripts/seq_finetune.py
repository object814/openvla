"""
seq_finetune.py

Script for sequentially fine-tuning OpenVLA models using LoRA. Modified from finetune.py

Run with:
    CUDA_VISIBLE_DEVICES=<GPU_IDS> torchrun <--standalone> --nnodes 1 --nproc-per-node $K seq_finetune.py \
        --data_root_dir <PATH/TO/DATASETS> \
        --dataset_names <DATASET1,DATASET2,...> \
        --run_root_dir <PATH/TO/RUNS> \
        --adapter_tmp_dir <PATH/TO/ADAPTER_TMP> \
        --batch_size <BATCH_SIZE> \
        --max_steps <MAX_STEPS> \
        --save_steps <SAVE_STEPS> \
        --learning_rate <LEARNING_RATE> \
        --grad_accumulation_steps <GRAD_ACCUMULATION_STEPS> \
        --shuffle_buffer_size <SHUFFLE_BUFFER_SIZE> \
        --wandb_project <WANDB_PROJECT> \
        --wandb_entity <WANDB_ENTITY> \
        --run_id_note <RUN_ID_NOTE>

Example:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc-per-node 2 seq_finetune.py \
        --data_root_dir /data/your_username/Datasets \
        --dataset_names libero_object_0,libero_object_1,libero_object_2,libero_object_3 \
        --run_root_dir /data/your_username/Experiments/runs \
        --adapter_tmp_dir /data/your_username/Experiments/adapter_tmp \
        --batch_size 4 \
        --max_steps 50000 \
        --save_steps 5000 \
        --learning_rate 1e-4 \
        --wandb_project openvla_seq_finetune \
        --wandb_entity your_wandb_entity \
        --run_id_note "experiment_note"

Note:
    - Exclude `--standalone` if your setup doesn't support it.
    - Adjust `--nproc-per-node` based on the number of GPUs you are using.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TRANSFORMERS_CACHE'] = '/data/zhouhy/Datasets/huggingface_cache' # Set a custom transformers cache directory (with openvla model inside) if needed
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import datetime

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false" # According to the HF team, this is needed to avoid issues with tokenizers

@dataclass
class FinetuneConfig:
    # Model and Paths
    vla_path: str = "openvla/openvla-7b"  # Path to OpenVLA model (on HuggingFace Hub)
    data_root_dir: str = "datasets/open-x-embodiment"  # Root directory for datasets
    dataset_names: str = "droid_wipe"  # Comma-separated list of dataset names
    run_root_dir: str = "runs"  # Directory to store logs and checkpoints
    adapter_tmp_dir: str = "adapter-tmp"  # Temp directory for LoRA weights

    # Fine-tuning Parameters
    batch_size: int = 2
    max_steps: int = 25_000
    save_steps: int = 12_500
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = False
    shuffle_buffer_size: int = 10_000
    save_latest_checkpoint_only: bool = False  # Save only the latest checkpoint

    # LoRA Arguments
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False  # Quantization can reduce memory but hurt performance

    # Tracking Parameters
    wandb_project: str = "openvla"
    wandb_entity: str = "your-wandb-entity" # object814-national-university-of-singapore
    run_id_note: Optional[str] = None  # Extra note for logging

    def __post_init__(self):
        # Convert paths to Path objects
        self.data_root_dir = Path(self.data_root_dir)
        self.run_root_dir = Path(self.run_root_dir)
        self.adapter_tmp_dir = Path(self.adapter_tmp_dir)
        # Parse dataset names into a list
        self.dataset_names = [name.strip() for name in self.dataset_names.split(",")]


@draccus.wrap()
def seq_finetune(cfg: FinetuneConfig) -> None:
    print(f"Sequentially fine-tuning OpenVLA Model {cfg.vla_path}")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Initialize W&B once for the entire run
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"run_{datetime.datetime.now().strftime('%Y%m%d')}")

    for dataset_name in cfg.dataset_names:
        print(f"Fine-tuning on `{dataset_name}`")

        # Configure Unique Experiment ID & Log Directory
        exp_id = (
            f"{dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
            f"+date-{datetime.datetime.now().strftime('%Y%m%d')}"
        )
        if cfg.use_lora:
            exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.use_quantization:
            exp_id += "+q-4bit"
        if cfg.run_id_note:
            exp_id += f"--{cfg.run_id_note}"
        if cfg.image_aug:
            exp_id += "--image_aug"

        # Start =>> Build Directories
        run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)

        # Quantization Config =>> only if LoRA fine-tuning
        quantization_config = None
        if cfg.use_quantization:
            assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            )

        # Load OpenVLA Processor and Model using HF AutoClasses
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
        if cfg.use_quantization:
            vla = prepare_model_for_kbit_training(vla)
        else:
            vla = vla.to(device_id)

        # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            if distributed_state.is_main_process:
                vla.print_trainable_parameters()

        # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
        vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

        # Create Optimizer =>> note that we default to a simple constant learning rate!
        trainable_params = [param for param in vla.parameters() if param.requires_grad]
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

        # Create Action Tokenizer
        action_tokenizer = ActionTokenizer(processor.tokenizer)

        # Load Fine-tuning Dataset
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        )
        vla_dataset = RLDSDataset(
            cfg.data_root_dir,
            dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )

        # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
        if distributed_state.is_main_process:
            save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

        # Create Collator and DataLoader
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )
        dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        )

        # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
        recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
        recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
        recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

        # Initialize step counter
        step_counter = 0

        # Train!
        # with tqdm.tqdm(total=cfg.max_steps, leave=True) as progress:
        with tqdm.tqdm(total=cfg.max_steps, disable=True) as progress:
            vla.train()
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(dataloader):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps

                # Backward pass
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Store recent train metrics
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss.item())

                # Compute gradient step index
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

                # Compute smoothened train metrics
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

                # Push Metrics to W&B (every 10 gradient steps)
                if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                    wandb.log(
                        {
                            f"{dataset_name}/train_loss": smoothened_loss,
                            f"{dataset_name}/action_accuracy": smoothened_action_accuracy,
                            f"{dataset_name}/l1_loss": smoothened_l1_loss,
                        },
                        step=step_counter,
                    )

                # Optimizer Step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update(cfg.grad_accumulation_steps)
                    step_counter += cfg.grad_accumulation_steps

                # Save Model Checkpoint
                if step_counter > 0 and step_counter % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {step_counter}")

                        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                        save_dir = adapter_dir if cfg.use_lora else run_dir

                        # Save Processor & Weights
                        processor.save_pretrained(run_dir)
                        vla.module.save_pretrained(save_dir)

                    # Wait for processor and adapter weights to be saved by main process
                    dist.barrier()

                    # Merge LoRA weights into model backbone for faster inference
                    #   =>> Note that merging is slow and can be done post-hoc to speed up training
                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        if distributed_state.is_main_process:
                            if cfg.save_latest_checkpoint_only:
                                # Overwrite latest checkpoint
                                merged_vla.save_pretrained(run_dir)
                                print(f"Saved Model Checkpoint for Step {step_counter} at: {run_dir}")
                            else:
                                # Prepare to save checkpoint in new directory
                                checkpoint_dir = run_dir / f"checkpoint-{step_counter}"
                                os.makedirs(checkpoint_dir, exist_ok=True)

                                # Save dataset statistics to new directory
                                save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                                # Save processor and model weights to new directory
                                processor.save_pretrained(checkpoint_dir)
                                merged_vla.save_pretrained(checkpoint_dir)

                                print(f"Saved Model Checkpoint for Step {step_counter} at: {checkpoint_dir}")

                    # Block on Main Process Checkpointing
                    dist.barrier()

                # Stop training when max_steps is reached
                if step_counter >= cfg.max_steps:
                    print(f"Max step {cfg.max_steps} reached for dataset {dataset_name}! Stopping training on this dataset...")
                    break

        # Update vla_path for the next dataset
        cfg.vla_path = str(run_dir)

        # Release CUDA memory
        torch.cuda.empty_cache()

    # Close W&B run
    if distributed_state.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    seq_finetune()


### The Following is the backup version of this script ###
# """
# seq_finetune.py

# Script for sequentially parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
# HuggingFace PEFT library for low-rank adaptation (LoRA).

# Sequentially fine-tunes OpenVLA on a number of datasets

# Changed from the original script: finetune.py

# Notes & Benchmarks:
#     - Requires PEFT (`pip install peft==0.11.1`)
#     - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
#         + One 48 GB GPU can fit a Batch Size of 12
#         + One 80 GB GPU can fit a Batch Size of 24

# Run with:
#     - [Single Node Multi-GPU (= $K) ]: torchrun --nnodes 1 --nproc-per-node $K vla-scripts/seq_finetune.py
#     - [Override Config Values]: torchrun --nnodes 1 --nproc-per-node $K vla-scripts/seq_finetune.py \
#                                     --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
#                                     --dataset_name <DATASET_NAME> \
#                                     --run_root_dir <PATH/TO/LOGS/DIR> \
#                                     ...
        
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc-per-node 4 vla-scripts/seq_finetune.py
# """

# import os
# os.environ['TRANSFORMERS_CACHE'] = '/data/zhouhy/Datasets/huggingface_cache'
# from collections import deque
# from dataclasses import dataclass, field
# from pathlib import Path
# import datetime

# import draccus
# import torch
# import torch.distributed as dist
# import tqdm
# from accelerate import PartialState
# from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
# from transformers.modeling_outputs import CausalLMOutputWithPast

# import wandb
# from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
# from prismatic.util.data_utils import PaddedCollatorForActionPrediction
# from prismatic.vla.action_tokenizer import ActionTokenizer
# from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
# from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# # Some Defaults
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # For NUS LinSLab server better performance
# # torch.set_num_threads(6)

# # # === Utilities ===
# # # fmt: off
# # def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
# #     """Gets image transform for the vision encoder."""
# #     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
# #     data_cfg["input_size"] = (3, input_size, input_size)
# #     return timm.data.create_transform(
# #         input_size=data_cfg["input_size"],
# #         interpolation=data_cfg["interpolation"],
# #         mean=data_cfg["mean"],
# #         std=data_cfg["std"],
# #         crop_pct=1.0,           # Set to 1.0 to disable cropping
# #         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
# #         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
# #     )
# #
# # # fmt: on


# @dataclass
# class FinetuneConfig:
#     # fmt: off
#     vla_path: str = "openvla/openvla-7b"                                                # Path to OpenVLA model (on HuggingFace Hub) to start with

#     # Directory Paths
#     data_root_dir: Path = Path("/data/zhouhy/Datasets/LIBERO_object_rlds")              # Path to parent directory for all RLDS datasets
#     dataset_names: list = field(default_factory=lambda: [
#         "libero_object_0",
#         "libero_object_1",
#         "libero_object_2",
#         "libero_object_3",
#         "libero_object_4",
#         "libero_object_5",
#         "libero_object_6",
#         "libero_object_7",
#         "libero_object_8",
#         "libero_object_9",
#     ])                                                                              # Name of fine-tuning datasets in sequence, they should be under `data_root_dir`
#     run_root_dir: Path = Path("/data/zhouhy/Datasets/openvla_seq_finetune_libero_object/data_dir")            # Path to directory to store logs & checkpoints
#     adapter_tmp_dir: Path = Path("/data/zhouhy/Datasets/openvla_seq_finetune_libero_object/adapter_temp")     # Temporary directory for LoRA weights before fusing

#     # Fine-tuning Parameters
#     batch_size: int = 2                                                                 # Fine-tuning batch size
#     max_steps: int = 25000                                                              # Max number of fine-tuning steps for each dataset (total steps = max_steps * len(dataset_names))
#     save_steps: int = 10000                                                             # Interval for checkpoint saving
#     learning_rate: float = 2e-5                                                         # Fine-tuning learning rate
#     grad_accumulation_steps: int = 1                                                    # Gradient accumulation steps
#     image_aug: bool = False                                                             # Whether to train with image augmentations
#     shuffle_buffer_size: int = 1000                                                    # Dataloader shuffle buffer size (can reduce if OOM)

#     # LoRA Arguments
#     use_lora: bool = True                                                               # Whether to use LoRA fine-tuning
#     lora_rank: int = 32                                                                 # Rank of LoRA weight matrix
#     lora_dropout: float = 0.0                                                           # Dropout applied to LoRA weights
#     use_quantization: bool = False                                                      # Whether to 4-bit quantize VLA for LoRA fine-tuning
#                                                                                         #   => CAUTION: Reduces memory but hurts performance

#     # Tracking Parameters
#     wandb_project: str = "openvla_seq_object"                                                  # Name of W&B project to log to (use default!)
#     wandb_entity: str = "object814-national-university-of-singapore"                    # Name of entity to log under

#     # fmt: on


# @draccus.wrap()
# def seq_finetune(cfg: FinetuneConfig) -> None:
#     print(f"Sequentially fine-tuning OpenVLA Model `{cfg.vla_path}`")

#     # Initialize W&B once for the entire run
#     if torch.distributed.get_rank() == 0:
#         wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name="seq_finetune")

#     for dataset_name in cfg.dataset_names:
#         print(f"Fine-tuning on `{dataset_name}`")

#         # [Validate] Ensure GPU Available & Set Device / Distributed Context
#         assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
#         distributed_state = PartialState()
#         torch.cuda.set_device(device_id := distributed_state.local_process_index)
#         torch.cuda.empty_cache()

#         # Configure Unique Experiment ID & Log Directory
#         exp_id = (
#             f"{dataset_name}"
#             f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
#             f"+lr-{cfg.learning_rate}"
#             f"+date-{datetime.datetime.now().strftime('%Y%m%d')}"
#         )
#         if cfg.use_lora:
#             exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
#         if cfg.use_quantization:
#             exp_id += "+q-4bit"

#         # Start =>> Build Directories
#         run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
#         os.makedirs(run_dir, exist_ok=True)

#         # Quantization Config =>> only if LoRA fine-tuning
#         quantization_config = None
#         if cfg.use_quantization:
#             assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
#             quantization_config = BitsAndBytesConfig(
#                 load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
#             )

#         # Load OpenVLA Processor and Model using HF AutoClasses
#         processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
#         vla = AutoModelForVision2Seq.from_pretrained(
#             cfg.vla_path,
#             torch_dtype=torch.bfloat16,
#             quantization_config=quantization_config,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#         )

#         # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
#         if cfg.use_quantization:
#             vla = prepare_model_for_kbit_training(vla)
#         else:
#             vla = vla.to(device_id)

#         # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
#         if cfg.use_lora:
#             lora_config = LoraConfig(
#                 r=cfg.lora_rank,
#                 lora_alpha=min(cfg.lora_rank, 16),
#                 lora_dropout=cfg.lora_dropout,
#                 target_modules="all-linear",
#                 init_lora_weights="gaussian",
#             )
#             vla = get_peft_model(vla, lora_config)
#             vla.print_trainable_parameters()

#         # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
#         vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

#         # Create Optimizer =>> note that we default to a simple constant learning rate!
#         trainable_params = [param for param in vla.parameters() if param.requires_grad]
#         optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

#         # Create Action Tokenizer
#         action_tokenizer = ActionTokenizer(processor.tokenizer)

#         # Load Fine-tuning Dataset
#         batch_transform = RLDSBatchTransform(
#             action_tokenizer,
#             processor.tokenizer,
#             image_transform=processor.image_processor.apply_transform,
#             prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
#         )
#         vla_dataset = RLDSDataset(
#             cfg.data_root_dir,
#             dataset_name,
#             batch_transform,
#             resize_resolution=tuple(vla.module.config.image_sizes),
#             shuffle_buffer_size=cfg.shuffle_buffer_size,
#             image_aug=cfg.image_aug,
#         )

#         # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
#         if distributed_state.is_main_process:
#             save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

#         # Create Collator and DataLoader
#         collator = PaddedCollatorForActionPrediction(
#             processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
#         )
#         dataloader = DataLoader(
#             vla_dataset,
#             batch_size=cfg.batch_size,
#             sampler=None,
#             collate_fn=collator,
#             num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
#         )

#         # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
#         recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
#         recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
#         recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

#         # Initialize step counter
#         step_counter = 0

#         # Train!
#         with tqdm.tqdm(total=cfg.max_steps, leave=True) as progress:
#             vla.train()
#             optimizer.zero_grad()
#             for batch_idx, batch in enumerate(dataloader):
#                 with torch.autocast("cuda", dtype=torch.bfloat16):
#                     output: CausalLMOutputWithPast = vla(
#                         input_ids=batch["input_ids"].to(device_id),
#                         attention_mask=batch["attention_mask"].to(device_id),
#                         pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
#                         labels=batch["labels"],
#                     )
#                     loss = output.loss

#                 # Normalize loss to account for gradient accumulation
#                 normalized_loss = loss / cfg.grad_accumulation_steps

#                 # Backward pass
#                 normalized_loss.backward()

#                 # Compute Accuracy and L1 Loss for Logging
#                 action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
#                 action_preds = action_logits.argmax(dim=2)
#                 action_gt = batch["labels"][:, 1:].to(action_preds.device)
#                 mask = action_gt > action_tokenizer.action_token_begin_idx

#                 # Compute Accuracy
#                 correct_preds = (action_preds == action_gt) & mask
#                 action_accuracy = correct_preds.sum().float() / mask.sum().float()

#                 # Compute L1 Loss on Predicted (Continuous) Actions
#                 continuous_actions_pred = torch.tensor(
#                     action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
#                 )
#                 continuous_actions_gt = torch.tensor(
#                     action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
#                 )
#                 action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

#                 # Store recent train metrics
#                 recent_losses.append(loss.item())
#                 recent_action_accuracies.append(action_accuracy.item())
#                 recent_l1_losses.append(action_l1_loss.item())

#                 # Compute gradient step index
#                 gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

#                 # Compute smoothened train metrics
#                 smoothened_loss = sum(recent_losses) / len(recent_losses)
#                 smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
#                 smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

#                 # Push Metrics to W&B (every 10 gradient steps)
#                 if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
#                     wandb.log(
#                         {
#                             f"{dataset_name}/train_loss": smoothened_loss,
#                             f"{dataset_name}/action_accuracy": smoothened_action_accuracy,
#                             f"{dataset_name}/l1_loss": smoothened_l1_loss,
#                         },
#                         step=gradient_step_idx,
#                     )

#                 # Optimizer Step
#                 if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
#                     optimizer.step()
#                     optimizer.zero_grad()
#                     progress.update(cfg.grad_accumulation_steps)
#                     step_counter += cfg.grad_accumulation_steps

#                 # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
#                 if gradient_step_idx > 0 and (gradient_step_idx+1) % cfg.save_steps == 0:
#                     if distributed_state.is_main_process:
#                         print(f"Saving Model Checkpoint for Step {gradient_step_idx+1}")

#                         # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
#                         save_dir = adapter_dir if cfg.use_lora else run_dir

#                         # Save Processor & Weights
#                         processor.save_pretrained(run_dir)
#                         vla.module.save_pretrained(save_dir)

#                     # Wait for processor and adapter weights to be saved by main process
#                     dist.barrier()

#                     # Merge LoRA weights into model backbone for faster inference
#                     #   =>> Note that merging is slow and can be done post-hoc to speed up training
#                     if cfg.use_lora:
#                         base_vla = AutoModelForVision2Seq.from_pretrained(
#                             cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
#                         )
#                         merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
#                         merged_vla = merged_vla.merge_and_unload()
#                         if distributed_state.is_main_process:
#                             merged_vla.save_pretrained(run_dir)

#                     # Block on Main Process Checkpointing
#                     dist.barrier()
                    
#                 if step_counter >= cfg.max_steps:
#                     break

#         # Update vla_path for the next dataset
#         cfg.vla_path = str(run_dir)

#         # release CUDA memory
#         torch.cuda.empty_cache()

# if __name__ == "__main__":
#     seq_finetune()
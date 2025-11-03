from typing import Union

import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache


class nnUNetTrainer_wandb(nnUNetTrainer):
    """
    nnU-Net Trainer with Weights & Biases (wandb) integration for real-time training monitoring.
    
    Usage:
        nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainer_wandb
    
    Make sure to install wandb first:
        pip install wandb
        wandb login  # authenticate once
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Initialize wandb
        try:
            import wandb
            self.wandb = wandb
            
            # Initialize wandb run
            dataset_name = dataset_json.get('name', f'Dataset{plans.get("dataset_name", "Unknown")}')
            run_name = f"{dataset_name}_{configuration}_fold{fold}"
            
            self.wandb.init(
                project="nnunet",
                name=run_name,
                config={
                    "dataset": dataset_name,
                    "configuration": configuration,
                    "fold": fold,
                    "initial_lr": self.initial_lr,
                    "weight_decay": self.weight_decay,
                    "batch_size": plans['configurations'][configuration]['batch_size'],
                    "num_epochs": self.num_epochs,
                },
                reinit=True
            )
            
            self.use_wandb = True
            self.print_to_log_file(f"Wandb initialized: {run_name}")
        except ImportError:
            self.wandb = None
            self.use_wandb = False
            self.print_to_log_file("WARNING: wandb not installed. Install with: pip install wandb")
        except Exception as e:
            self.wandb = None
            self.use_wandb = False
            self.print_to_log_file(f"WARNING: Failed to initialize wandb: {e}")

    def log_to_wandb(self, metrics: dict, step: int = None):
        """Helper method to log metrics to wandb"""
        if not self.use_wandb:
            return
        
        if step is None:
            step = self.current_epoch
        
        self.wandb.log(metrics, step=step)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.use_wandb:
            self.log_to_wandb({
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }, step=self.current_epoch)

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        
        if self.use_wandb and len(self.logger.my_fantastic_logging['train_losses']) > 0:
            train_loss = self.logger.my_fantastic_logging['train_losses'][-1]
            self.log_to_wandb({
                "train/loss": train_loss,
            }, step=self.current_epoch)

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        
        if not self.use_wandb:
            return
        
        metrics = {}
        
        if len(self.logger.my_fantastic_logging['val_losses']) > 0:
            metrics["val/loss"] = self.logger.my_fantastic_logging['val_losses'][-1]
        
        if len(self.logger.my_fantastic_logging['mean_fg_dice']) > 0:
            metrics["val/mean_fg_dice"] = self.logger.my_fantastic_logging['mean_fg_dice'][-1]
        
        if len(self.logger.my_fantastic_logging['ema_fg_dice']) > 0:
            metrics["val/ema_fg_dice"] = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
        
        if len(self.logger.my_fantastic_logging['dice_per_class_or_region']) > 0:
            dice_per_class = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
            if isinstance(dice_per_class, (list, tuple)):
                for idx, dice_val in enumerate(dice_per_class):
                    metrics[f"val/dice_class_{idx}"] = dice_val
            elif isinstance(dice_per_class, dict):
                for key, dice_val in dice_per_class.items():
                    metrics[f"val/dice_{key}"] = dice_val
        
        if metrics:
            self.log_to_wandb(metrics, step=self.current_epoch)

    def finish_online_evaluation(self):
        super().finish_online_evaluation()
        
        if self.use_wandb:
            if len(self.logger.my_fantastic_logging['epoch_end_timestamps']) > 0 and \
               len(self.logger.my_fantastic_logging['epoch_start_timestamps']) > 0:
                epoch_durations = [
                    end - start 
                    for end, start in zip(
                        self.logger.my_fantastic_logging['epoch_end_timestamps'],
                        self.logger.my_fantastic_logging['epoch_start_timestamps']
                    )
                ]
                if epoch_durations:
                    avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
                    self.log_to_wandb({
                        "timing/avg_epoch_time_seconds": avg_epoch_time
                    })

    def finish_training(self):
        super().finish_training()
        
        if self.use_wandb:
            try:
                self.wandb.finish()
                self.print_to_log_file("Wandb run finished successfully")
            except Exception as e:
                self.print_to_log_file(f"WARNING: Error finishing wandb run: {e}")

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """Override to handle wandb resuming"""
        super().load_checkpoint(filename_or_checkpoint)
        
        if self.use_wandb:
            self.log_to_wandb({
                "checkpoint/loaded_epoch": self.current_epoch
            }, step=self.current_epoch)


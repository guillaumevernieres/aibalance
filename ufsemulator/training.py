"""
Training application for UfsEmulatorFFNN model
Handles data loading, training, validation, and model saving.
Uses YAML configuration files for easy parameter management.
"""

import argparse
import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Union, cast
from typing import Any, List, Optional

from .model import create_ufs_emulator_ffnn, UfsEmulatorFFNN
from .data import create_training_data_from_netcdf


class UfsEmulatorTrainer:
    """Training class for UfsEmulatorFFNN model with distributed support."""

    def __init__(self, config: Dict, rank: int = 0, world_size: int = 1):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration dictionary
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

        # Set CPU threading for non-distributed training
        if not self.is_distributed and not torch.cuda.is_available():
            num_threads = config.get('num_threads', None)
            if num_threads is None:
                # Try to get from environment or use all available CPUs
                num_threads = int(
                    os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 1)
                )
            torch.set_num_threads(num_threads)
            if self.rank == 0:
                print(f"PyTorch CPU threads: {torch.get_num_threads()}")

        # Setup device
        if self.is_distributed:
            # Use LOCAL_RANK for device assignment if available
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            if torch.cuda.is_available():
                # Ensure we don't exceed available GPUs
                local_rank = local_rank % torch.cuda.device_count()
                self.device = torch.device(f'cuda:{local_rank}')
                torch.cuda.set_device(local_rank)
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(
                'cuda' if (torch.cuda.is_available()
                           and config.get('use_cuda', True))
                else 'cpu'
            )

        if self.rank == 0:
            print(f"Device: {self.device}")
            if self.is_distributed:
                print(f"Distributed: {world_size} processes")

        # Determine model dimensions from variable configuration
        input_size, output_size = self._get_model_dimensions()

        # Initialize model
        hidden_layers = config['model'].get('hidden_layers', 2)
        self.model: Union[UfsEmulatorFFNN, DDP] = create_ufs_emulator_ffnn(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            output_size=output_size,
            hidden_layers=hidden_layers
        ).to(self.device)

        # Wrap model for distributed training
        if self.is_distributed:
            device_ids = [rank] if torch.cuda.is_available() else None
            self.model = DDP(self.model, device_ids=device_ids)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize loss function
        self.criterion = self._create_loss_function()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'jacobian_frobenius_norm': [],
            'jacobian_spectral_norm': [],
            'jacobian_stability': [],
            'jacobian_metrics_history': []  # Store full metrics dictionaries
        }

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.config['training']['optimizer']

        if opt_config['type'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        elif opt_config['type'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")

    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_type = self.config['training'].get('loss_function', 'mse')

        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        scheduler_config = self.config['training'].get('scheduler', None)

        if scheduler_config is None:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1000,
                gamma=1.0
            )  # No-op scheduler

        if scheduler_config['type'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_config['type'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        else:
            raise ValueError(
                f"Unknown scheduler type: {scheduler_config['type']}"
            )

    def load_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare training data with distributed support.

        Args:
            data_path: Path to data file (.npz, .pt, or .nc)

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Handle NetCDF files by converting them first
        if data_path.endswith('.nc'):
            if self.rank == 0:
                print("Converting NetCDF to training format...")
                processed_file = str(Path(data_path).with_suffix('.npz'))

                # Use data preparation module
                create_training_data_from_netcdf(
                    data_path,
                    self.config,
                    processed_file
                )

            # Synchronize processes if distributed
            if self.is_distributed:
                dist.barrier()

            data_path = str(Path(data_path).with_suffix('.npz'))

        # Load processed data
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            inputs = torch.FloatTensor(data['inputs'])
            targets = torch.FloatTensor(data['targets'])

            # Use saved normalization stats if available
            if 'input_mean' in data and 'input_std' in data:
                input_mean = torch.tensor(
                    data['input_mean'], dtype=torch.float32
                )
                input_std = torch.tensor(
                    data['input_std'], dtype=torch.float32
                )
            else:
                # Compute normalization statistics (distributed if needed)
                input_mean, input_std = self._compute_distributed_stats(inputs)

        elif data_path.endswith('.pt'):
            data = torch.load(data_path)
            inputs = data['inputs']
            targets = data['targets']

            # Use saved normalization stats if available
            if 'input_mean' in data and 'input_std' in data:
                input_mean = data['input_mean']
                input_std = data['input_std']
            else:
                # Compute normalization statistics (distributed if needed)
                input_mean, input_std = self._compute_distributed_stats(inputs)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        # Removed verbose data shape logging

        # Prevent division by zero
        input_std = torch.where(
            input_std > 1e-6,
            input_std,
            torch.ones_like(input_std)
        )

        # Initialize model normalization
        if hasattr(self.model, 'module'):  # DDP wrapped model
            ddp_model = cast(DDP, self.model)
            assert hasattr(ddp_model.module, 'init_norm')
            ddp_model.module.init_norm(input_mean, input_std)
        else:
            emulator_model = cast(UfsEmulatorFFNN, self.model)
            assert hasattr(emulator_model, 'init_norm')
            emulator_model.init_norm(input_mean, input_std)

        # Save normalization once (not per checkpoint)
        if self.rank == 0:
            output_dir = Path(self.config['output']['model_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            norm_path = output_dir / "normalization.pt"

            if hasattr(self.model, 'module'):
                ddp_model = cast(DDP, self.model)
                ddp_model.module.save_norm(str(norm_path))
            else:
                emulator_model = cast(UfsEmulatorFFNN, self.model)
                emulator_model.save_norm(str(norm_path))

        # Create dataset
        dataset = TensorDataset(inputs, targets)

        # Split into train/validation
        val_size = int(len(dataset) * self.config['data']['validation_split'])
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        # Create distributed samplers if needed
        if self.is_distributed:
            train_sampler: Optional[DistributedSampler] = DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank
            )
            val_sampler: Optional[DistributedSampler] = DistributedSampler(
                val_dataset, num_replicas=self.world_size, rank=self.rank
            )
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.config['data'].get('num_workers', 0)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config['data'].get('num_workers', 0)
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Fix shape mismatch: squeeze model output to match target shape
            if outputs.dim() > targets.dim():
                outputs = outputs.squeeze()

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Removed batch-level progress logging for cleaner output
        return total_loss / num_batches

    def _get_prediction_sample(self, val_loader: DataLoader,
                               max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a sample of predictions vs targets for plotting.

        Args:
            val_loader: Validation data loader
            max_samples: Maximum number of samples to collect

        Returns:
            Tuple of (predictions, targets) as numpy arrays
        """
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for inputs, batch_targets in val_loader:
                if len(predictions) >= max_samples:
                    break

                inputs = inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                outputs = self.model(inputs)

                # Fix shape mismatch: squeeze model output to match target shape
                if outputs.dim() > batch_targets.dim():
                    outputs = outputs.squeeze()

                predictions.extend(outputs.cpu().numpy().flatten())
                targets.extend(batch_targets.cpu().numpy().flatten())

        return np.array(predictions[:max_samples]), np.array(targets[:max_samples])

    def compute_jacobian_metrics(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Compute Jacobian convergence metrics on a sample of validation data.

        For data assimilation, the Jacobian (∂output/∂input) is crucial as it
        represents the sensitivity of ice concentration to input variables.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with Jacobian metrics:
            - frobenius_norm: Frobenius norm of the Jacobian
            - spectral_norm: Largest singular value of the Jacobian
            - stability: Measure of Jacobian stability/conditioning
        """
        self.model.eval()

        # Sample a subset of validation data for Jacobian computation
        sample_size = min(100, len(val_loader.dataset))  # Limit for efficiency
        jacobians = []

        with torch.enable_grad():  # Need gradients for Jacobian
            for i, (inputs, _) in enumerate(val_loader):
                if i * val_loader.batch_size >= sample_size:
                    break

                inputs = inputs.to(self.device)
                batch_size = inputs.shape[0]

                # Compute Jacobian for each sample in batch
                max_samples = min(batch_size,
                                  sample_size - i * val_loader.batch_size)
                for j in range(max_samples):
                    sample_input = inputs[j:j+1].requires_grad_(True)

                    # Forward pass
                    output = self.model(sample_input)

                    # Compute gradients (Jacobian row)
                    grad_outputs = torch.ones_like(output)
                    jacobian_row = torch.autograd.grad(
                        outputs=output,
                        inputs=sample_input,
                        grad_outputs=grad_outputs,
                        create_graph=False,
                        retain_graph=False
                    )[0]

                    jacobians.append(
                        jacobian_row.detach().cpu().numpy().flatten())

        if not jacobians:
            return {
                'frobenius_norm': 0.0,
                'spectral_norm': 0.0,
                'stability': 0.0
            }

        # Stack all Jacobian rows into matrix
        jacobian_matrix = np.vstack(jacobians)

        # Compute different meaningful metrics for better visualization
        # 1. Frobenius norm: overall magnitude of all gradients
        frobenius_norm = np.linalg.norm(jacobian_matrix, 'fro')

        # 2. Mean absolute gradient: average sensitivity across features
        mean_abs_gradient = np.mean(np.abs(jacobian_matrix))

        # 3. Compute SVD for spectral norm and stability
        try:
            U, s, Vt = np.linalg.svd(jacobian_matrix, full_matrices=False)
            # Use actual spectral norm (largest singular value)
            spectral_norm = s[0] if len(s) > 0 else 0.0

            # Stability metric: ratio of largest to smallest singular value
            stability = (s[0] / s[-1] if len(s) > 0 and s[-1] > 1e-12
                         else 1e6)

        except np.linalg.LinAlgError:
            spectral_norm = 0.0
            stability = 1e6

        # Additional metrics for enhanced visualization
        max_gradient = np.max(np.abs(jacobian_matrix)) if jacobian_matrix.size > 0 else 0.0
        gradient_std = np.std(jacobian_matrix) if jacobian_matrix.size > 0 else 0.0

        # Feature-wise sensitivity (most significant jacobian per feature)
        feature_sensitivity = np.mean(np.abs(jacobian_matrix), axis=0) if jacobian_matrix.size > 0 else np.array([])
        most_sensitive_feature = np.argmax(feature_sensitivity) if len(feature_sensitivity) > 0 else 0
        max_feature_sensitivity = np.max(feature_sensitivity) if len(feature_sensitivity) > 0 else 0.0

        return {
            'frobenius_norm': float(frobenius_norm),
            'spectral_norm': float(mean_abs_gradient),
            'stability': float(stability),
            'max_gradient': float(max_gradient),
            'gradient_std': float(gradient_std),
            'feature_sensitivity': feature_sensitivity,
            'most_sensitive_feature': int(most_sensitive_feature),
            'max_feature_sensitivity': float(max_feature_sensitivity)
        }

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                # Fix shape mismatch: squeeze model output to match target shape
                if outputs.dim() > targets.dim():
                    outputs = outputs.squeeze()

                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("Starting training...")
        start_time = time.time()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training'].get('early_stopping_patience', 10)

        # Convergence tracking
        convergence_tolerance = self.config['training'].get(
            'convergence_tolerance', 1e-6)
        convergence_window = self.config['training'].get(
            'convergence_window', 5)
        min_epochs = self.config['training'].get('min_epochs', 10)
        recent_losses = []

        # Jacobian convergence tracking
        jacobian_convergence_tolerance = self.config['training'].get(
            'jacobian_convergence_tolerance', 1e-4)
        jacobian_convergence_window = self.config['training'].get(
            'jacobian_convergence_window', 5)
        min_epochs_jacobian = self.config['training'].get(
            'min_epochs_jacobian', 20)
        use_jacobian_stopping = self.config['training'].get(
            'use_jacobian_stopping', True)
        recent_jacobian_norms = []
        recent_jacobian_stability = []
        recent_jacobian_mean_abs_gradients = []

        for epoch in range(self.config['training']['epochs']):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Compute Jacobian metrics (every few epochs to avoid overhead)
            track_jacobian = self.config['training'].get(
                'track_jacobian', True)
            jacobian_freq = self.config['training'].get('jacobian_freq', 5)

            if track_jacobian and (epoch + 1) % jacobian_freq == 0:
                jacobian_metrics = self.compute_jacobian_metrics(val_loader)

                # Store both individual metrics and full dictionary
                self.history['jacobian_frobenius_norm'].append(
                    jacobian_metrics['frobenius_norm'])
                self.history['jacobian_spectral_norm'].append(
                    jacobian_metrics['spectral_norm'])
                self.history['jacobian_stability'].append(
                    jacobian_metrics['stability'])
                self.history['jacobian_metrics_history'].append(jacobian_metrics)

                # Track recent Jacobian metrics for convergence
                recent_jacobian_norms.append(
                    jacobian_metrics['frobenius_norm'])
                recent_jacobian_stability.append(
                    jacobian_metrics['stability'])
                recent_jacobian_mean_abs_gradients.append(
                    jacobian_metrics['spectral_norm'])  # spectral_norm contains mean_abs_gradient

                # Keep only recent values for convergence check
                if len(recent_jacobian_norms) > jacobian_convergence_window:
                    recent_jacobian_norms.pop(0)
                window_limit = jacobian_convergence_window
                if len(recent_jacobian_stability) > window_limit:
                    recent_jacobian_stability.pop(0)
                if len(recent_jacobian_mean_abs_gradients) > jacobian_convergence_window:
                    recent_jacobian_mean_abs_gradients.pop(0)
            else:
                # Pad with previous values to keep arrays aligned
                if track_jacobian:
                    if self.history['jacobian_frobenius_norm']:
                        self.history['jacobian_frobenius_norm'].append(
                            self.history['jacobian_frobenius_norm'][-1])
                        self.history['jacobian_spectral_norm'].append(
                            self.history['jacobian_spectral_norm'][-1])
                        self.history['jacobian_stability'].append(
                            self.history['jacobian_stability'][-1])
                    else:
                        # First epoch, initialize with zeros
                        self.history['jacobian_frobenius_norm'].append(0.0)
                        self.history['jacobian_spectral_norm'].append(0.0)
                        self.history['jacobian_stability'].append(0.0)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Track recent losses for convergence detection
            recent_losses.append(val_loss)
            if len(recent_losses) > convergence_window:
                recent_losses.pop(0)

            epoch_time = time.time() - epoch_start

            # Early stopping and convergence checks
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1

            # Check for convergence (loss plateau)
            converged = False
            jacobian_converged = False

            if (epoch + 1 >= min_epochs and
                    len(recent_losses) == convergence_window):

                # Check if loss has plateaued (relative change < tolerance)
                max_loss = max(recent_losses)
                min_loss = min(recent_losses)
                if max_loss > 0:
                    relative_change = (max_loss - min_loss) / max_loss
                    if relative_change < convergence_tolerance:
                        converged = True
                        print(f'Loss convergence detected: loss plateau '
                              f'(relative change: {relative_change:.2e} < '
                              f'{convergence_tolerance:.2e})')

            # Check for Jacobian convergence - prioritize mean absolute gradient convergence
            if (use_jacobian_stopping and track_jacobian and
                    (epoch + 1) >= min_epochs_jacobian and
                    len(recent_jacobian_mean_abs_gradients) >= jacobian_convergence_window):

                # Primary criterion: Check Jacobian mean absolute gradient convergence
                max_mean_grad = max(recent_jacobian_mean_abs_gradients)
                min_mean_grad = min(recent_jacobian_mean_abs_gradients)
                if max_mean_grad > 0:
                    mean_grad_relative_change = ((max_mean_grad - min_mean_grad) /
                                                max_mean_grad)
                    tolerance = jacobian_convergence_tolerance
                    if mean_grad_relative_change < tolerance:
                        jacobian_converged = True
                        print(f'Jacobian convergence detected: '
                              f'Mean absolute gradient plateau '
                              f'(relative change: '
                              f'{mean_grad_relative_change:.2e} < {tolerance:.2e})')

                # Secondary criterion: Check Jacobian Frobenius norm convergence
                if not jacobian_converged and len(recent_jacobian_norms) >= jacobian_convergence_window:
                    max_jac_norm = max(recent_jacobian_norms)
                    min_jac_norm = min(recent_jacobian_norms)
                    if max_jac_norm > 0:
                        jac_relative_change = ((max_jac_norm - min_jac_norm) /
                                               max_jac_norm)
                        if jac_relative_change < tolerance:
                            jacobian_converged = True
                            print(f'Jacobian convergence detected: '
                                  f'Frobenius norm plateau '
                                  f'(relative change: '
                                  f'{jac_relative_change:.2e} < {tolerance:.2e})')

            # Only show epoch progress every 10 epochs or at key points
            if (epoch + 1) % 10 == 0 or epoch == 0:
                # Include Jacobian mean abs gradient info if available
                jacobian_info = ""
                if track_jacobian and recent_jacobian_mean_abs_gradients:
                    current_mean_grad = recent_jacobian_mean_abs_gradients[-1]
                    jacobian_info = f', Jac MAG: {current_mean_grad:.4f}'
                print(f'Epoch {epoch+1}/{self.config["training"]["epochs"]} - '
                      f'Train: {train_loss:.6f}, Val: {val_loss:.6f}{jacobian_info}')

            # Stop training based on different criteria - prioritize Jacobian convergence
            if jacobian_converged:
                print(f'Training stopped due to Jacobian convergence after {epoch+1} epochs')
                #break
            elif patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)')
                #break
            elif converged:
                print(f'Training converged (loss plateau) after {epoch+1} epochs')
                #break

            # Save periodic checkpoint and update plots
            save_interval = self.config['training'].get('save_interval', 10)
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

                # Get a sample of predictions vs targets for plotting
                sample_predictions, sample_targets = self._get_prediction_sample(val_loader)

                # Update training plots at the same interval
                self.plot_training_history(
                    output_dir=self.config['output']['model_dir'],
                    train_loss=self.history['train_loss'],
                    val_loss=self.history['val_loss'],
                    learning_rate=self.history['learning_rate'],
                    jacobian_metrics_history=self.history['jacobian_metrics_history'],
                    predictions=sample_predictions,
                    targets=sample_targets
                )

        total_time = time.time() - start_time
        print(f'Training completed in {total_time:.1f}s')

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        # Get model state dict handling DDP wrapping
        if hasattr(self.model, 'module'):  # DDP wrapped
            model_state = cast(DDP, self.model).module.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }

        output_dir = Path(self.config['output']['model_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, output_dir / filename)

        # Only show checkpoint save for best model or every 50 epochs
        #if 'best_model' in filename or int(filename.split('_')[-1].split('.')[0]) % 50 == 0:
        #    print(f'Saved: {filename}')

    def save_model(self, model_path: str) -> None:
        """
        Save model state dict only (for testing compatibility).

        Args:
            model_path: Path to save the model
        """
        model_state = self.model.state_dict()
        if hasattr(self.model, 'module'):  # DDP wrapped model
            ddp_model = cast(DDP, self.model)
            model_state = ddp_model.module.state_dict()

        torch.save(model_state, model_path)

        print(f'Model saved: {model_path}')

    def plot_training_history(self, output_dir, train_loss, val_loss,
                             learning_rate, jacobian_metrics_history=None,
                             predictions=None, targets=None):
        """
        Plot comprehensive training history with enhanced Jacobian diagnostics.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))

        # 1. Training and validation loss (top-left)
        axes[0, 0].plot(train_loss, label='Training Loss', color='blue')
        axes[0, 0].plot(val_loss, label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        # Check if log scale is requested
        use_log_scale = self.config.get('training', {}).get(
            'use_log_scale_loss', True)
        if use_log_scale:
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_ylabel('Loss (log scale)')
        else:
            axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Learning rate (top-middle)
        axes[0, 1].plot(learning_rate, color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        # Extract Jacobian metrics if available
        if jacobian_metrics_history:
            frobenius_norms = [m.get('frobenius_norm', 0) for m in jacobian_metrics_history]
            spectral_norms = [m.get('spectral_norm', 0) for m in jacobian_metrics_history]
            stability_vals = [m.get('stability', 0) for m in jacobian_metrics_history]
            max_gradients = [m.get('max_gradient', 0) for m in jacobian_metrics_history]
            gradient_stds = [m.get('gradient_std', 0) for m in jacobian_metrics_history]
            max_sensitivities = [m.get('max_feature_sensitivity', 0) for m in jacobian_metrics_history]
        else:
            frobenius_norms = spectral_norms = stability_vals = []
            max_gradients = gradient_stds = max_sensitivities = []

        # 3. Jacobian Frobenius norm (top-right)
        if len(frobenius_norms) > 0:
            axes[0, 2].plot(frobenius_norms, color='purple')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Frobenius Norm')
            axes[0, 2].set_title('Jacobian Frobenius Norm')
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'No Jacobian Data', ha='center', va='center')
            axes[0, 2].set_title('Jacobian Frobenius Norm')

        # 4. Jacobian Spectral norm (top-far-right)
        if len(spectral_norms) > 0:
            axes[0, 3].plot(spectral_norms, color='orange')
            axes[0, 3].set_xlabel('Epoch')
            axes[0, 3].set_ylabel('Mean Abs Gradient')
            axes[0, 3].set_title('Jacobian Mean Abs Gradient')
            axes[0, 3].grid(True, alpha=0.3)
        else:
            axes[0, 3].text(0.5, 0.5, 'No Jacobian Data', ha='center', va='center')
            axes[0, 3].set_title('Jacobian Mean Abs Gradient')

        # 5. Maximum Gradient (bottom-left)
        if len(max_gradients) > 0:
            axes[1, 0].plot(max_gradients, color='red')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Max |Gradient|')
            axes[1, 0].set_title('Maximum Jacobian Element')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Jacobian Data', ha='center', va='center')
            axes[1, 0].set_title('Maximum Jacobian Element')

        # 6. Target vs Inference scatter (bottom-middle)
        if predictions is not None and targets is not None:
            valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
            if np.any(valid_mask):
                pred_valid = predictions[valid_mask]
                targ_valid = targets[valid_mask]

                axes[1, 1].scatter(targ_valid, pred_valid, alpha=0.6, s=1)
                min_val = min(np.min(targ_valid), np.min(pred_valid))
                max_val = max(np.max(targ_valid), np.max(pred_valid))
                axes[1, 1].plot([min_val, max_val], [min_val, max_val],
                                'r--', alpha=0.8, label='Perfect Prediction')

                rmse = np.sqrt(np.mean((pred_valid - targ_valid) ** 2))
                bias = np.mean(pred_valid - targ_valid)
                axes[1, 1].text(0.05, 0.95, f'RMSE: {rmse:.4f}\nBias: {bias:.4f}',
                                transform=axes[1, 1].transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                axes[1, 1].set_xlabel('Target Ice Concentration')
                axes[1, 1].set_ylabel('Predicted Ice Concentration')
                axes[1, 1].set_title('Target vs Prediction')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Valid Data', ha='center', va='center')
                axes[1, 1].set_title('Target vs Prediction')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Data', ha='center', va='center')
            axes[1, 1].set_title('Target vs Prediction')

        # 7. Most Sensitive Feature (bottom-middle-right)
        if len(max_sensitivities) > 0:
            axes[1, 2].plot(max_sensitivities, color='green')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Max Feature Sensitivity')
            axes[1, 2].set_title('Most Significant Jacobian')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Jacobian Data', ha='center', va='center')
            axes[1, 2].set_title('Most Significant Jacobian')

        # 8. Jacobian Stability (bottom-right)
        if len(stability_vals) > 0:
            axes[1, 3].semilogy(stability_vals, color='brown')
            axes[1, 3].set_xlabel('Epoch')
            axes[1, 3].set_ylabel('Condition Number')
            axes[1, 3].set_title('Jacobian Stability')
            axes[1, 3].grid(True, alpha=0.3)
        else:
            axes[1, 3].text(0.5, 0.5, 'No Jacobian Data', ha='center', va='center')
            axes[1, 3].set_title('Jacobian Stability')

        plt.suptitle('UfsEmulatorFFNN Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot
        output_path = Path(output_dir) / 'training_history.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_convergence_diagnostics(self, train_loss, val_loss, predictions, targets, output_dir):
        """
        Plot training/validation loss and target vs inference with stats.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Check if log scale is requested
        use_log_scale = self.config.get('training', {}).get(
            'use_log_scale_loss', True)

        # Panel 1: Training loss
        axes[0, 0].plot(train_loss, label='Train Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        if use_log_scale:
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_ylabel('Loss (log scale)')
        else:
            axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        # Panel 2: Validation loss
        axes[0, 1].plot(val_loss, label='Validation Loss', color='orange')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        if use_log_scale:
            axes[0, 1].set_yscale('log')
            axes[0, 1].set_ylabel('Loss (log scale)')
        else:
            axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()

        # Panel 3: Target vs Inference scatter
        axes[1, 0].scatter(targets, predictions, s=2, alpha=0.5)
        axes[1, 0].set_title('Target vs Inference')
        axes[1, 0].set_xlabel('Target (Ground Truth)')
        axes[1, 0].set_ylabel('Inference (Prediction)')
        # Compute stats
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        bias = np.mean(predictions - targets)
        axes[1, 0].text(0.05, 0.95, f'RMSE: {rmse:.4f}\nBias: {bias:.4f}',
                        transform=axes[1, 0].transAxes,
                        fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        # Panel 4: Histogram of prediction errors
        error = predictions - targets
        axes[1, 1].hist(error, bins=50, color='gray', alpha=0.7)
        axes[1, 1].set_title('Prediction Error Histogram')
        axes[1, 1].set_xlabel('Prediction - Target')
        axes[1, 1].set_ylabel('Count')

        plt.tight_layout()
        fig.savefig(f'{output_dir}/convergence_diagnostics.png', dpi=150)
        plt.close(fig)

    def _compute_distributed_stats(
            self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and std statistics across distributed processes.
        Replicates the C++ MPI_Allreduce functionality.

        Args:
            inputs: Local input tensor

        Returns:
            Tuple of (global_mean, global_std)
        """
        if not self.is_distributed:
            # Single process case
            return inputs.mean(dim=0), inputs.std(dim=0)

        # Compute local statistics
        local_sum = torch.sum(inputs, dim=0)
        local_sq_sum = torch.sum(torch.pow(inputs, 2), dim=0)
        local_count = torch.tensor(inputs.size(0), dtype=torch.float32)

        # All-reduce across processes (equivalent to MPI_Allreduce)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        # Compute global mean and std
        global_mean = local_sum / local_count
        global_var = (local_sq_sum / local_count) - torch.pow(global_mean, 2)
        global_std = torch.sqrt(global_var)

        return global_mean, global_std

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint and resume training from saved state.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if hasattr(self.model, 'module'):  # DDP wrapped model
            ddp_model = cast(DDP, self.model)
            ddp_model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            emulator_model = cast(UfsEmulatorFFNN, self.model)
            emulator_model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training history
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        # Load configuration (for validation)
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            # Verify model architecture matches
            if (saved_config['model']['input_size'] != self.config['model']['input_size'] or
                saved_config['model']['output_size'] != self.config['model']['output_size']):
                raise ValueError("Model architecture mismatch between checkpoint and current config")

        if self.rank == 0:
            print(f"Resuming from epoch {len(self.history['train_loss'])}")

    def load_best_model(self, model_dir: str = None) -> None:
        """
        Load the best saved model for inference or resumed training.

        Args:
            model_dir: Directory containing the best_model.pt file
        """
        if model_dir is None:
            model_dir = self.config['output']['model_dir']

        best_model_path = Path(model_dir) / 'best_model.pt'

        if not best_model_path.exists():
            raise FileNotFoundError(f"Best model not found: {best_model_path}")

        # Load checkpoint (best_model.pt contains full checkpoint)
        self.load_checkpoint(str(best_model_path))

        if self.rank == 0:
            print("Best model loaded")

    def _get_model_dimensions(self) -> Tuple[int, int]:
        """
        Determine input and output sizes from variable configuration.

        Returns:
            Tuple of (input_size, output_size)
        """
        variables_config = self.config.get('variables', {})

        # Get input variables
        input_variables = variables_config.get('input_variables', [
            'sst', 'sss', 'tair', 'tsfc', 'hi', 'hs', 'sice',
            'strocnx', 'strocny', 'strairx', 'strairy', 'qref',
            'flwdn', 'fswdn'
        ])

        # Apply max_input_features limit if specified
        max_input_features = variables_config.get('max_input_features')
        if max_input_features is not None and max_input_features > 0:
            input_variables = input_variables[:max_input_features]

        # Get output variables
        output_variables = variables_config.get('output_variables', ['aice'])

        input_size = len(input_variables)
        output_size = len(output_variables)

        if self.rank == 0:
            print(f"Model configuration: {input_size} inputs, "
                  f"{output_size} outputs")
            print(f"Input variables: {input_variables}")
            print(f"Output variables: {output_variables}")

        # Update config for compatibility
        self.config['model']['input_size'] = input_size
        self.config['model']['output_size'] = output_size

        return input_size, output_size

def create_sample_data(config: Dict) -> str:
    """
    Create sample training data for testing.

    Args:
        config: Configuration dictionary

    Returns:
        Path to saved data file
    """
    print("Creating sample data...")

    input_size = config['model']['input_size']
    num_samples = config['data'].get('num_samples', 10000)

    # Generate synthetic data matching the NetCDF structure
    np.random.seed(42)

    # Create correlated input features (tair, tsfc, sst, sss, hs, hi, sice)
    inputs = np.random.randn(num_samples, input_size).astype(np.float32)

    # Create realistic ranges for each variable
    # Air temperature: -40 to +20 C
    inputs[:, 0] = inputs[:, 0] * 15 + (-10)
    # Surface temperature: -35 to +25 C
    inputs[:, 1] = inputs[:, 1] * 15 + (-5)
    # Sea surface temperature: -2 to +30 C
    inputs[:, 2] = inputs[:, 2] * 8 + 14
    # Sea surface salinity: 30 to 37 psu
    inputs[:, 3] = inputs[:, 3] * 1.5 + 33.5
    # Snow thickness: 0 to 2 m
    inputs[:, 4] = np.abs(inputs[:, 4]) * 0.5
    # Ice thickness: 0 to 5 m
    inputs[:, 5] = np.abs(inputs[:, 5]) * 1.2
    # Ice salinity: 0 to 20 psu
    inputs[:, 6] = np.abs(inputs[:, 6]) * 4 + 6

    # Create synthetic ice concentration target (0-1)
    # Based on temperature and existing ice
    temp_factor = np.maximum(0, (-inputs[:, 0] + 10) / 30)  # Colder = more ice
    ice_factor = inputs[:, 5] / 5.0  # More thickness = higher concentration
    noise = np.random.randn(num_samples) * 0.1

    targets = np.clip(temp_factor * 0.7 + ice_factor * 0.3 + noise, 0, 1)
    targets = targets.astype(np.float32)

    # Save data
    data_path = Path(config['data']['data_path'])
    data_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(data_path, inputs=inputs, targets=targets)
    print(f"Sample data saved: {data_path}")

    return str(data_path)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        result = yaml.safe_load(f)
        if not isinstance(result, dict):
            raise ValueError(
                f"Configuration file {config_path} must contain a dictionary"
            )
        return result


def create_default_config() -> Dict:
    """Create default training configuration."""
    return {
        "model": {
            "input_size": 8,  # Updated: 7 features + latitude
            "hidden_size": 16,
            "output_size": 1  # Updated to match C++ (1 output: aice)
        },
        "domain": {
            "pole": "north",
            "clean_data": True
        },
        "data": {
            "data_path": "data/sample_data.npz",
            "validation_split": 0.2,
            "num_workers": 0,
            "num_samples": 10000
        },
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 1e-5
            },
            "scheduler": {
                "type": "step",
                "step_size": 30,
                "gamma": 0.5
            },
            "loss_function": "mse",
            "early_stopping_patience": 15,
            "log_interval": 50,
            "save_interval": 20,
            "track_jacobian": True,
            "jacobian_freq": 5,
            "convergence_tolerance": 1e-6,
            "convergence_window": 5,
            "min_epochs": 10,
            "use_jacobian_stopping": True,
            "jacobian_convergence_tolerance": 1e-3,  # More sensitive for mean abs gradient
            "jacobian_convergence_window": 10,  # Longer window for more stability
            "min_epochs_jacobian": 30  # Wait longer before checking Jacobian convergence
        },
        "output": {
            "model_dir": "models/"
        },
        "use_cuda": True
    }


def setup_distributed(rank: int = None, world_size: int = None) -> Tuple[int, int]:
    """
    Initialize distributed training for HPC environments.

    Returns:
        Tuple of (rank, world_size)
    """
    # HPC systems often set these environment variables
    if 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', rank))

        # SLURM sets node list, extract master node
        node_list = os.environ.get('SLURM_NODELIST', 'localhost')

        # Parse SLURM node list format
        if '[' in node_list:
            # Handle node ranges like "node[01-04]" or "node[01,03-05]"
            base = node_list.split('[')[0]
            # Get first node number
            ranges = node_list.split('[')[1].split(']')[0]
            first_range = ranges.split(',')[0]
            if '-' in first_range:
                first_num = first_range.split('-')[0]
            else:
                first_num = first_range
            master_node = base + first_num
        else:
            # Simple comma-separated list
            master_node = node_list.split(',')[0]

        os.environ['MASTER_ADDR'] = master_node
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

        print(f"[Rank {rank}] SLURM setup: {master_node}:{os.environ['MASTER_PORT']}, "
              f"local_rank={local_rank}")

    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun or manual distributed setup
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Ensure MASTER_ADDR and MASTER_PORT are set
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'

    else:
        # Single process setup
        if rank is None:
            rank = 0
        if world_size is None:
            world_size = 1
        local_rank = 0
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

    # Initialize the process group
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        try:
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=30)  # Longer timeout for HPC
            )

            if rank == 0:
                print(f"Distributed training initialized: {backend} backend, "
                      f"{world_size} processes")
                print(f"Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

        except Exception as e:
            print(f"[Rank {rank}] Failed to initialize distributed training: {e}")
            raise
    else:
        if rank == 0:
            print("Single-process training (no distribution)")

    return rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_distributed(rank: int, world_size: int, config: Dict[str, Any],
                      data_path: str, restart_checkpoint: str = None,
                      restart_from_best: bool = False) -> None:
    """
    Distributed training function.

    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Training configuration
        data_path: Path to training data
        restart_checkpoint: Path to checkpoint file for restart
        restart_from_best: Whether to restart from best_model.pt
    """
    # Setup distributed training - now returns actual rank/world_size
    rank, world_size = setup_distributed(rank, world_size)

    try:
        # Initialize trainer with distributed settings
        trainer = UfsEmulatorTrainer(config, rank=rank, world_size=world_size)

        # Handle restart options
        if restart_checkpoint:
            trainer.load_checkpoint(restart_checkpoint)
        elif restart_from_best:
            trainer.load_best_model()

        # Load data
        train_loader, val_loader = trainer.load_data(data_path)

        # Train model
        trainer.train(train_loader, val_loader)

        # Plot training history (only on rank 0)
        if rank == 0:
            trainer.plot_training_history(
                output_dir=trainer.config['output']['model_dir'],
                train_loss=trainer.history['train_loss'],
                val_loss=trainer.history['val_loss'],
                learning_rate=trainer.history['learning_rate'],
                jacobian_metrics_history=(
                    trainer.history['jacobian_metrics_history']
                ),
                predictions=None,
                targets=None
            )
            print("Training completed!")

    finally:
        # Clean up
        if world_size > 1:
            cleanup_distributed()


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train UfsEmulatorFFNN model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    parser.add_argument('--create-data', action='store_true',
                        help='Create sample training data')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Override data path (.npz, .pt, or .nc)')
    parser.add_argument('--netcdf-file', type=str, default=None,
                        help='NetCDF file to convert to training data')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Total processes for distributed training')
    parser.add_argument('--no-distributed', action='store_true',
                        help='Disable distributed training even in '
                             'SLURM/MPI environment')
    parser.add_argument('--restart-from-best', action='store_true',
                        help='Restart training from best_model.pt checkpoint')
    parser.add_argument('--restart-from-checkpoint', type=str, default=None,
                        help='Restart training from specific checkpoint file')

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        print("Using default configuration")

    # Override data path if provided
    if args.data_path:
        config['data']['data_path'] = args.data_path
    elif args.netcdf_file:
        config['data']['data_path'] = args.netcdf_file

    # Create sample data if requested
    if args.create_data:
        data_path = create_sample_data(config)
        config['data']['data_path'] = data_path

    # Check if data exists
    data_file = config['data']['data_path']
    if not Path(data_file).exists():
        print(f"Data file not found: {data_file}")
        if data_file.endswith('.nc'):
            print("NetCDF file will be processed during training")
        else:
            print("Use --create-data to generate sample data")
            print("Or provide a NetCDF file with --netcdf-file")
            return

    # Distributed training - check for HPC environment
    world_size = args.world_size
    rank = args.local_rank

    # Override with HPC environment variables if present (unless disabled)
    if not args.no_distributed:
        if 'SLURM_PROCID' in os.environ and 'SLURM_NPROCS' in os.environ:
            # SLURM automatically sets these - let setup_distributed handle it
            rank = None  # Will be determined by setup_distributed
            world_size = None  # Will be determined by setup_distributed
            distributed_detected = int(os.environ['SLURM_NPROCS']) > 1
        elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # torchrun or manual setup
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            distributed_detected = world_size > 1
        else:
            distributed_detected = world_size > 1
    else:
        # Force single-process mode
        rank = 0
        world_size = 1
        distributed_detected = False

    if distributed_detected:
        print("Distributed training detected")
        # Call distributed training function
        train_distributed(
            rank, world_size, config, config['data']['data_path'],
            args.restart_from_checkpoint, args.restart_from_best
        )
    else:
        # Initialize trainer
        trainer = UfsEmulatorTrainer(config)

        # Handle restart options
        if args.restart_from_checkpoint:
            trainer.load_checkpoint(args.restart_from_checkpoint)
        elif args.restart_from_best:
            trainer.load_best_model()

        # Load data
        data_path = config['data']['data_path']
        train_loader, val_loader = trainer.load_data(data_path)

        # Train model
        trainer.train(train_loader, val_loader)

        # Plot training history
        trainer.plot_training_history(
            output_dir=trainer.config['output']['model_dir'],
            train_loss=trainer.history['train_loss'],
            val_loss=trainer.history['val_loss'],
            learning_rate=trainer.history['learning_rate'],
            jacobian_metrics_history=trainer.history['jacobian_metrics_history'],
            predictions=None,
            targets=None
        )

        print("Training completed!")

"""
Simplified training application for UfsEmulatorFFNN model.
Non-distributed version for single-node training with threading support.
"""

import argparse
import os
import time
import numpy as np
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List

from .model import create_ufs_emulator_ffnn, UfsEmulatorFFNN
from .data import create_training_data_from_netcdf


class UfsEmulatorTrainer:
    """Simple training class for UfsEmulatorFFNN model."""

    def __init__(self, config: Dict):
        """Initialize trainer with configuration."""
        self.config = config

        # Set CPU threading
        num_threads = config.get('num_threads')
        if num_threads is None:
            num_threads = int(os.environ.get('OMP_NUM_THREADS',
                                             os.cpu_count() or 1))
        torch.set_num_threads(num_threads)
        print(f"PyTorch CPU threads: {torch.get_num_threads()}")

        # Setup device
        self.device = torch.device(
            'cuda' if (torch.cuda.is_available() and
                      config.get('use_cuda', True))
            else 'cpu'
        )
        print(f"Device: {self.device}")

        # Get model dimensions from config
        input_size, output_size = self._get_model_dimensions()

        # Initialize model
        hidden_layers = config['model'].get('hidden_layers', 2)
        self.model = create_ufs_emulator_ffnn(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            output_size=output_size,
            hidden_layers=hidden_layers
        ).to(self.device)

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
            'jacobian_metrics_history': []
        }

    def _get_model_dimensions(self) -> Tuple[int, int]:
        """Determine input/output sizes from variable configuration."""
        variables_config = self.config.get('variables', {})

        input_variables = variables_config.get('input_variables', [
            'sst', 'sss', 'tair', 'tsfc', 'hi', 'hs', 'sice',
            'strocnx', 'strocny', 'strairx', 'strairy', 'qref',
            'flwdn', 'fswdn'
        ])

        max_input_features = variables_config.get('max_input_features')
        if max_input_features and max_input_features > 0:
            input_variables = input_variables[:max_input_features]

        output_variables = variables_config.get('output_variables', ['aice'])

        input_size = len(input_variables)
        output_size = len(output_variables)

        print(f"Model: {input_size} inputs, {output_size} outputs")
        print(f"Inputs: {input_variables}")
        print(f"Outputs: {output_variables}")

        self.config['model']['input_size'] = input_size
        self.config['model']['output_size'] = output_size

        return input_size, output_size

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
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
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")

    def _create_loss_function(self) -> nn.Module:
        """Create loss function from config."""
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
        scheduler_config = self.config['training'].get('scheduler')

        if scheduler_config is None:
            # No-op scheduler
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1000, gamma=1.0
            )

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
            raise ValueError(f"Unknown scheduler: {scheduler_config['type']}")

    def load_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare training data."""
        # Convert NetCDF to npz if needed
        if data_path.endswith('.nc'):
            print("Converting NetCDF to training format...")
            processed_file = str(Path(data_path).with_suffix('.npz'))
            create_training_data_from_netcdf(
                data_path, self.config, processed_file
            )
            data_path = processed_file

        # Load data
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            inputs = torch.FloatTensor(data['inputs'])
            targets = torch.FloatTensor(data['targets'])
            input_mean = torch.tensor(data['input_mean'], dtype=torch.float32)
            input_std = torch.tensor(data['input_std'], dtype=torch.float32)
        elif data_path.endswith('.pt'):
            data = torch.load(data_path)
            inputs = data['inputs']
            targets = data['targets']
            input_mean = data['input_mean']
            input_std = data['input_std']
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        # Prevent division by zero
        input_std = torch.where(
            input_std > 1e-6, input_std, torch.ones_like(input_std)
        )

        # Initialize model normalization
        self.model.init_norm(input_mean, input_std)

        # Save normalization
        output_dir = Path(self.config['output']['model_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_norm(str(output_dir / "normalization.pt"))

        # Create dataset
        dataset = TensorDataset(inputs, targets)

        # Split train/validation
        val_split = self.config['data']['validation_split']
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        # Create data loaders
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['data'].get('num_workers', 0)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Handle shape mismatch
            if outputs.dim() > targets.dim():
                outputs = outputs.squeeze()

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                if outputs.dim() > targets.dim():
                    outputs = outputs.squeeze()

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def compute_jacobian_metrics(
            self, val_loader: DataLoader) -> Dict[str, float]:
        """Compute Jacobian metrics for data assimilation."""
        self.model.eval()
        sample_size = min(100, len(val_loader.dataset))
        jacobians = []

        with torch.enable_grad():
            for i, (inputs, _) in enumerate(val_loader):
                if i * val_loader.batch_size >= sample_size:
                    break

                inputs = inputs.to(self.device)
                batch_size = inputs.shape[0]
                max_samples = min(
                    batch_size, sample_size - i * val_loader.batch_size
                )

                for j in range(max_samples):
                    sample_input = inputs[j:j+1].requires_grad_(True)
                    output = self.model(sample_input)

                    grad_outputs = torch.ones_like(output)
                    jacobian_row = torch.autograd.grad(
                        outputs=output,
                        inputs=sample_input,
                        grad_outputs=grad_outputs,
                        create_graph=False,
                        retain_graph=False
                    )[0]

                    jacobians.append(
                        jacobian_row.detach().cpu().numpy().flatten()
                    )

        if not jacobians:
            return {
                'frobenius_norm': 0.0,
                'mean_abs_gradient': 0.0,
                'stability': 0.0,
                'max_gradient': 0.0
            }

        jacobian_matrix = np.vstack(jacobians)
        frobenius_norm = np.linalg.norm(jacobian_matrix, 'fro')
        mean_abs_gradient = np.mean(np.abs(jacobian_matrix))
        max_gradient = np.max(np.abs(jacobian_matrix))

        try:
            _, s, _ = np.linalg.svd(jacobian_matrix, full_matrices=False)
            stability = (s[0] / s[-1] if len(s) > 0 and s[-1] > 1e-12
                        else 1e6)
        except np.linalg.LinAlgError:
            stability = 1e6

        return {
            'frobenius_norm': float(frobenius_norm),
            'mean_abs_gradient': float(mean_abs_gradient),
            'stability': float(stability),
            'max_gradient': float(max_gradient)
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Main training loop."""
        print("Starting training...")
        start_time = time.time()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training'].get('early_stopping_patience', 10)

        for epoch in range(self.config['training']['epochs']):
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step()

            # Compute Jacobian metrics
            track_jacobian = self.config['training'].get(
                'track_jacobian', True
            )
            jacobian_freq = self.config['training'].get('jacobian_freq', 5)

            if track_jacobian and (epoch + 1) % jacobian_freq == 0:
                jacobian_metrics = self.compute_jacobian_metrics(val_loader)
                self.history['jacobian_metrics_history'].append(
                    jacobian_metrics
                )

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1

            # Progress display
            if (epoch + 1) % 10 == 0 or epoch == 0:
                jac_info = ""
                if (track_jacobian and
                        self.history['jacobian_metrics_history']):
                    jac = self.history['jacobian_metrics_history'][-1]
                    jac_info = f", Jac: {jac['mean_abs_gradient']:.4f}"
                print(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}'
                      f' - Train: {train_loss:.6f}, '
                      f'Val: {val_loss:.6f}{jac_info}')

            # Early stopping check
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break

            # Save periodic checkpoints and plots
            save_interval = self.config['training'].get('save_interval', 10)
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
                self.plot_training_history()

        total_time = time.time() - start_time
        print(f'Training completed in {total_time:.1f}s')

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }

        output_dir = Path(self.config['output']['model_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_dir / filename)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        print(f"Resumed from epoch {len(self.history['train_loss'])}")

    def plot_training_history(self) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss plot
        axes[0, 0].plot(self.history['train_loss'],
                       label='Training Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'],
                       label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        use_log_scale = self.config.get('training', {}).get(
            'use_log_scale_loss', True
        )
        if use_log_scale:
            axes[0, 0].set_yscale('log')

        # Learning rate
        axes[0, 1].plot(self.history['learning_rate'], color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        # Jacobian metrics
        if self.history['jacobian_metrics_history']:
            jac_history = self.history['jacobian_metrics_history']
            mean_grads = [m['mean_abs_gradient'] for m in jac_history]
            frob_norms = [m['frobenius_norm'] for m in jac_history]

            axes[1, 0].plot(mean_grads, color='orange')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Mean Abs Gradient')
            axes[1, 0].set_title('Jacobian Mean Abs Gradient')
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(frob_norms, color='purple')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Frobenius Norm')
            axes[1, 1].set_title('Jacobian Frobenius Norm')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Jacobian Data',
                           ha='center', va='center')
            axes[1, 1].text(0.5, 0.5, 'No Jacobian Data',
                           ha='center', va='center')

        plt.suptitle('UfsEmulatorFFNN Training History',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = Path(self.config['output']['model_dir'])
        output_path = output_path / 'training_history.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        result = yaml.safe_load(f)
        if not isinstance(result, dict):
            raise ValueError(
                f"Config file {config_path} must contain a dictionary"
            )
        return result


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train UfsEmulatorFFNN model'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--data-path', type=str,
                       help='Override data path (.npz, .pt, or .nc)')
    parser.add_argument('--restart-from-checkpoint', type=str,
                       help='Restart from checkpoint file')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override data path if provided
    if args.data_path:
        config['data']['data_path'] = args.data_path

    # Check data exists
    data_file = config['data']['data_path']
    if not Path(data_file).exists():
        print(f"Data file not found: {data_file}")
        if data_file.endswith('.nc'):
            print("NetCDF file will be processed during training")
        else:
            print("Error: Data file does not exist")
            return

    # Initialize trainer
    trainer = UfsEmulatorTrainer(config)

    # Load checkpoint if requested
    if args.restart_from_checkpoint:
        trainer.load_checkpoint(args.restart_from_checkpoint)

    # Load data and train
    train_loader, val_loader = trainer.load_data(config['data']['data_path'])
    trainer.train(train_loader, val_loader)
    trainer.plot_training_history()

    print("Training completed!")


if __name__ == "__main__":
    main()

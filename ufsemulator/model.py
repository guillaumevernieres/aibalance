"""
UfsEmulatorFFNN PyTorch Model in Python
FFNN to emulate UFS and provide Jacobians
for balance operators.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from pathlib import Path
from typing import List, Dict


class UfsEmulatorFFNN(nn.Module):
    """
    Feed Forward NN emulator for UFS.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 hidden_layers: int = 2, kernel_size: int = 1,
                 stride: int = 1):
        """
        Initialize the UfsEmulatorFFNN model.
        """
        super(UfsEmulatorFFNN, self).__init__()

        print(
            f"Starting UfsEmulatorFFNN constructor: {input_size} -> "
            f"{hidden_layers}x{hidden_size} -> {output_size}"
        )

        # Build dynamic network based on hidden_layers
        layers = []

        # First layer: input -> hidden
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.GELU())

        # Additional hidden layers: hidden -> hidden
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        # Final layer: hidden -> output (no activation - linear output)
        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

        # Register mean and std as buffers (non-trainable parameters)
        self.register_buffer('input_mean', torch.full((input_size,), 0.0))
        self.register_buffer('input_std', torch.full((input_size,), 1.0))
        self.register_buffer('output_mean', torch.full((output_size,), 0.0))
        self.register_buffer('output_std', torch.full((output_size,), 1.0))

        # Type annotations for buffers (for mypy)
        self.input_mean: torch.Tensor
        self.input_std: torch.Tensor
        self.output_mean: torch.Tensor
        self.output_std: torch.Tensor

        # TorchScript-serializable metadata for IO names and arbitrary meta
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_levels: List[int] = []
        self.output_levels: List[int] = []
        self.meta: Dict[str, str] = {}

        # Compute total number of parameters (degrees of freedom)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total degrees of freedom (parameters): {total_params}")

        print("End UfsEmulatorFFNN constructor")

    def set_io_names(self, inputs: List[str], outputs: List[str]) -> None:
        """
        Set input/output variable names. These will be serialized with TorchScript.
        """
        self.input_names = list(inputs)
        self.output_names = list(outputs)

    def set_io_levels(self, input_levels: List[int], output_levels: List[int]) -> None:
        """
        Set input/output vertical level indices. These will be serialized with TorchScript.
        """
        self.input_levels = list(input_levels)
        self.output_levels = list(output_levels)

    def set_metadata(self, kv: Dict[str, str]) -> None:
        """
        Set arbitrary string metadata to be serialized with TorchScript.
        """
        self.meta = dict(kv)

    def init_norm(self, input_mean: torch.Tensor, input_std: torch.Tensor,
                  output_mean: torch.Tensor, output_std: torch.Tensor) -> None:
        """
        Initialize normalization parameters for both inputs and outputs.
        """
        self.input_mean.data = input_mean.clone()
        self.input_std.data = input_std.clone()
        self.output_mean.data = output_mean.clone()
        self.output_std.data = output_std.clone()

    def save_norm(self, model_filename: str) -> None:
        """
        Save normalization parameters to file (both input and output).
        """
        file_path = Path(model_filename)
        path = file_path.parent

        moments = {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std
        }
        norm_path = path / "normalization.pt"
        torch.save(moments, norm_path)
        print(f"Saved normalization to: {norm_path}")

    def load_norm(self, model_filename: str) -> None:
        """
        Load normalization parameters from file (both input and output).
        """
        file_path = Path(model_filename)
        path = file_path.parent

        norm_path = path / "normalization.pt"
        moments = torch.load(norm_path)

        self.input_mean.data = moments['input_mean']
        self.input_std.data = moments['input_std']
        self.output_mean.data = moments['output_mean']
        self.output_std.data = moments['output_std']

        print(f"Loaded normalization from: {norm_path}")

    def init_weights(self) -> None:
        """
        Initialize weights using Xavier normal initialization.
        """
        for module in self.network:
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
        print("Initialized weights with Xavier normal distribution")

    @torch.jit.export
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize physical input to standardized space.
        x_normalized = (x - mean) / std
        """
        return (x - self.input_mean) / self.input_std

    @torch.jit.export
    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """
        Denormalize network output to physical space.
        y_physical = y * std + mean
        """
        return y * self.output_std + self.output_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pure network.
        EXPECTS NORMALIZED INPUT, RETURNS NORMALIZED OUTPUT.

        For end-to-end prediction with physical inputs/outputs, use predict() instead.
        """
        return self.network(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end prediction: physical input -> physical output.
        This is the recommended method for inference.

        Args:
            x: Input in physical space [batch_size, input_size]

        Returns:
            Output in physical space [batch_size, output_size]
        """
        x_norm = self.normalize_input(x)
        y_norm = self.forward(x_norm)
        y_phys = self.denormalize_output(y_norm)
        return y_phys

    @torch.jit.export
    def jac(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian in NORMALIZED space: ∂y_norm/∂x_norm.

        This is the cleanest representation with no chain rule confusion.
        The input x should be in physical space and will be normalized internally.

        Args:
            x: Input in physical space [batch_size, input_size]

        Returns:
            Jacobian in normalized space [batch_size, output_size, input_size]
        """
        # Normalize input
        x_norm = self.normalize_input(x)
        
        batch_size = x_norm.shape[0]
        output_size = self.network[-1].out_features
        input_size = x_norm.shape[1]

        # Compute Jacobian column by column
        jacobians = []
        for i in range(output_size):
            # Create a fresh tensor with gradients enabled for each output
            x_norm_copy = x_norm.detach().requires_grad_(True)
            
            # Forward pass through pure network
            y_norm = self.forward(x_norm_copy)
            
            # Create gradient output vector (select i-th output)
            grad_outputs = torch.zeros_like(y_norm)
            grad_outputs[:, i] = 1.0
            
            # Compute gradient using backward pass
            y_norm.backward(grad_outputs)
            
            if x_norm_copy.grad is None:
                raise RuntimeError("Gradients not computed properly")
            
            jacobians.append(x_norm_copy.grad.clone())

        return torch.stack(jacobians, dim=1)

    @torch.jit.export
    def jac_physical(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian in PHYSICAL space: ∂y_phys/∂x_phys.

        Uses chain rule to transform from normalized Jacobian:
        ∂y_phys/∂x_phys = (output_std / input_std) * ∂y_norm/∂x_norm

        Args:
            x: Input in physical space [batch_size, input_size]

        Returns:
            Jacobian in physical space [batch_size, output_size, input_size]
        """
        # Get normalized Jacobian: ∂y_norm/∂x_norm
        jac_norm = self.jac(x)

        # Apply chain rule scaling
        # Broadcasting:
        #   jac_norm: [batch, output, input]
        #   output_std: [output] -> [1, output, 1]
        #   input_std: [input] -> [1, 1, input]
        output_std_expanded = self.output_std.view(1, -1, 1)
        input_std_expanded = self.input_std.view(1, 1, -1)

        jac_phys = jac_norm * output_std_expanded / input_std_expanded
        return jac_phys

    def jac_norm(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute Frobenius norm of Jacobian (placeholder implementation).
        """
        return torch.tensor(0.0)

    def save_model(self, model_filename: str) -> None:
        """
        Save the entire model to file.
        """
        torch.save(self.state_dict(), model_filename)
        print(f"Saved model to: {model_filename}")

    def load_model(self, model_filename: str) -> None:
        """
        Load the model from file.
        """
        self.load_state_dict(torch.load(model_filename))
        print(f"Loaded model from: {model_filename}")

        for name, param in self.named_parameters():
            print(f"Parameter name: {name}, Size: {param.size()}")

        for name, buffer in self.named_buffers():
            print(f"Buffer name: {name}, Size: {buffer.size()}")
            print(f"       values: {buffer}")

    def save_torchscript(self, script_filename: str) -> None:
        """
        Export and save the model as a TorchScript module.
        The scripted module will include parameters and registered buffers
        (e.g., input_mean and input_std).
        """
        # Ensure the module is in eval mode for export
        was_training = self.training
        self.eval()
        try:
            scripted = torch.jit.script(self)
            torch.jit.save(scripted, script_filename)
            print(f"Saved TorchScript model to: {script_filename}")
        finally:
            # Restore original training mode
            if was_training:
                self.train()


def create_ufs_emulator_ffnn(
        input_size: int, hidden_size: int, output_size: int,
        hidden_layers: int = 2
) -> UfsEmulatorFFNN:
    """Factory to create and initialize a UfsEmulatorFFNN model."""
    model = UfsEmulatorFFNN(
        input_size,
        hidden_size,
        output_size,
        hidden_layers,
    )
    model.init_weights()
    return model


if __name__ == "__main__":
    print("Testing UfsEmulatorFFNN model...")
    model = create_ufs_emulator_ffnn(
        input_size=4,
        hidden_size=10,
        output_size=2,
    )
    # Set IO names and metadata example
    model.set_io_names(["u", "v", "t", "q"], ["du", "dv"])
    model.set_metadata({"version": "1.0"})

    # Create test input in physical space
    x = torch.randn(1, 4)
    print(f"Input (physical): {x}")

    # Initialize normalization (using identity for this test)
    input_mean = torch.zeros(4)
    input_std = torch.ones(4)
    output_mean = torch.zeros(2)
    output_std = torch.ones(2)
    model.init_norm(input_mean, input_std, output_mean, output_std)

    # Test normalization methods
    x_norm = model.normalize_input(x)
    print(f"Input (normalized): {x_norm}")

    # Test forward (expects normalized input)
    y_norm = model.forward(x_norm)
    print(f"Output from forward() [normalized]: {y_norm}")

    # Test predict (end-to-end with physical input/output)
    y_phys = model.predict(x)
    print(f"Output from predict() [physical]: {y_phys}")

    # Test Jacobian in normalized space
    jac_norm = model.jac(x)
    print(f"Jacobian (normalized space) shape: {jac_norm.shape}")
    print(f"Jacobian (normalized): {jac_norm}")

    # Test Jacobian in physical space
    jac_phys = model.jac_physical(x)
    print(f"Jacobian (physical space) shape: {jac_phys.shape}")
    print(f"Jacobian (physical): {jac_phys}")

    # Example TorchScript export
    model.save_torchscript("ufs_emulator_ffnn.pt")
    print("UfsEmulatorFFNN model test completed successfully!")


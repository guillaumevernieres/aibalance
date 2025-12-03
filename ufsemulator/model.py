"""
UfsEmulatorFFNN PyTorch Model in Python
FFNN to emulate UFS and provide Jacobians
for balance operators.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from pathlib import Path


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
        layers.append(nn.ReLU())

        # Additional hidden layers: hidden -> hidden
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Final layer: hidden -> output (no activation - linear output)
        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

        # Register mean and std as buffers (non-trainable parameters)
        self.register_buffer('input_mean', torch.full((input_size,), 0.0))
        self.register_buffer('input_std', torch.full((input_size,), 1.0))

        # Type annotations for buffers (for mypy)
        self.input_mean: torch.Tensor
        self.input_std: torch.Tensor

        # Compute total number of parameters (degrees of freedom)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total degrees of freedom (parameters): {total_params}")

        print("End UfsEmulatorFFNN constructor")

    def init_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """
        Initialize normalization parameters.
        """
        self.input_mean.data = mean.clone()
        self.input_std.data = std.clone()

    def save_norm(self, model_filename: str) -> None:
        """
        Save normalization parameters to file.
        """
        file_path = Path(model_filename)
        path = file_path.parent

        moments = [self.input_mean, self.input_std]
        norm_path = path / "normalization.pt"
        torch.save(moments, norm_path)
        print(f"Saved normalization to: {norm_path}")

    def load_norm(self, model_filename: str) -> None:
        """
        Load normalization parameters from file.
        """
        file_path = Path(model_filename)
        path = file_path.parent

        norm_path = path / "normalization.pt"
        moments = torch.load(norm_path)
        self.input_mean.data = moments[0]
        self.input_std.data = moments[1]
        print(f"Loaded normalization from: {norm_path}")

    def init_weights(self) -> None:
        """
        Initialize weights using Xavier normal initialization.
        """
        for module in self.network:
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
        print("Initialized weights with Xavier normal distribution")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        # Normalize the input
        x = (x - self.input_mean) / self.input_std
        return self.network(x)

    def jac(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian (dout/dx) using automatic differentiation.
        Returns [batch_size, output_size, input_size].
        """
        x_input = x.clone().detach().requires_grad_(True)
        y = self.forward(x_input)
        output_size = y.shape[1]

        jacobians = []
        for i in range(output_size):
            if x_input.grad is not None:
                x_input.grad.zero_()
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, i] = 1.0
            y.backward(grad_outputs, retain_graph=True)
            if x_input.grad is None:
                raise RuntimeError("Gradients not computed properly")
            jacobians.append(x_input.grad.clone())

        return torch.stack(jacobians, dim=1)

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
    x = torch.randn(1, 4)
    print(f"Input: {x}")
    mean = torch.zeros(4)
    std = torch.ones(4)
    model.init_norm(mean, std)
    output = model.forward(x)
    print(f"Output: {output}")
    jac = model.jac(x)
    print(f"Jacobian: {jac}")
    print("UfsEmulatorFFNN model test completed successfully!")

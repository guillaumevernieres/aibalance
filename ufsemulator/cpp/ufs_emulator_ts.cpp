#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

/**
 * Simple demonstration of using TorchScript UFS Emulator with jac_physical() method.
 *
 * Usage: ./ufs_emulator_ts <model.ts>
 */

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.ts>\n";
        std::cerr << "\nDemonstrate TorchScript model with jac_physical() method.\n";
        return 1;
    }

    std::string model_path = argv[1];

    std::cout << "========================================\n";
    std::cout << "UFS Emulator TorchScript Demo\n";
    std::cout << "========================================\n\n";

    // Load model
    std::cout << "Loading model: " << model_path << std::endl;
    torch::jit::script::Module model;
    model = torch::jit::load(model_path);
    model.eval();
    std::cout << "✓ Model loaded successfully\n" << std::endl;

    // Create test input (physical space)
    // Example: [sst, aice, tsfc, qref] for a single point
    torch::Tensor input = torch::tensor({{16.0, 0.0, 285.0, 0.003}});

    std::cout << "Test Input (physical space):\n";
    std::cout << "  Shape: " << input.sizes() << "\n";
    std::cout << "  Values: " << input << "\n" << std::endl;

    // Call jac_physical() method
    std::cout << "Computing Jacobian using jac_physical()...\n";
    torch::Tensor jacobian;
    try {
        auto method = model.get_method("jac_physical");
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        jacobian = method(inputs).toTensor();
        std::cout << "✓ Jacobian computed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error calling jac_physical(): " << e.what() << std::endl;
        return 1;
    }

    // Display Jacobian
    std::cout << "Jacobian (∂y_phys/∂x_phys):\n";
    std::cout << "  Shape: " << jacobian.sizes() << "\n";
    std::cout << "  Format: [batch, output_size, input_size]\n\n";

    // Extract dimensions
    int64_t batch_size = jacobian.size(0);
    int64_t output_size = jacobian.size(1);
    int64_t input_size = jacobian.size(2);

    // Print Jacobian matrix
    for (int64_t b = 0; b < batch_size; ++b) {
        std::cout << "  Batch " << b << ":\n";
        for (int64_t out = 0; out < output_size; ++out) {
            std::cout << "    Output[" << out << "]: ";
            for (int64_t in = 0; in < input_size; ++in) {
                double val = jacobian[b][out][in].item<double>();
                std::cout << std::setw(10) << std::fixed << std::setprecision(6) << val;
                if (in < input_size - 1) std::cout << "  ";
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "Demo completed successfully!\n";
    std::cout << "========================================\n";

    return 0;
}

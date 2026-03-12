/**
 * Test executable for ufs_emulator_ts - validates C++ Jacobian against Python reference
 *
 * Usage: ufs_emulator_ts_test <model.ts> <input.pt> <reference_jacobian.pt> <tolerance>
 */

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

struct ComparisonResult {
    bool passed;
    double max_abs_error;
    double mean_abs_error;
    double max_rel_error;
    int num_elements;
    int num_passed;
    int num_failed;
};

ComparisonResult compare_tensors(const torch::Tensor& a, const torch::Tensor& b, double tol) {
    ComparisonResult result;

    // Check shapes match
    if (a.sizes() != b.sizes()) {
        std::cerr << "Shape mismatch: " << a.sizes() << " vs " << b.sizes() << std::endl;
        result.passed = false;
        return result;
    }

    // Flatten for easier comparison
    auto a_flat = a.flatten();
    auto b_flat = b.flatten();

    // Compute errors
    auto abs_diff = torch::abs(a_flat - b_flat);
    auto rel_diff = abs_diff / (torch::abs(b_flat) + 1e-10);

    result.max_abs_error = abs_diff.max().item<double>();
    result.mean_abs_error = abs_diff.mean().item<double>();
    result.max_rel_error = rel_diff.max().item<double>();
    result.num_elements = a_flat.size(0);

    // Count passed/failed elements
    auto passed_mask = abs_diff <= tol;
    result.num_passed = passed_mask.sum().item<int>();
    result.num_failed = result.num_elements - result.num_passed;

    result.passed = (result.num_failed == 0);

    return result;
}

void print_result(const ComparisonResult& result, double tolerance) {
    std::cout << "\n=== Comparison Results ===" << std::endl;
    std::cout << "Total elements: " << result.num_elements << std::endl;
    std::cout << "Passed: " << result.num_passed << " ("
              << (100.0 * result.num_passed / result.num_elements) << "%)" << std::endl;
    std::cout << "Failed: " << result.num_failed << " ("
              << (100.0 * result.num_failed / result.num_elements) << "%)" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Max absolute error: " << result.max_abs_error << std::endl;
    std::cout << "Mean absolute error: " << result.mean_abs_error << std::endl;
    std::cout << "Max relative error: " << result.max_rel_error << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;
    std::cout << "\n=== Test " << (result.passed ? "PASSED" : "FAILED") << " ===" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.ts> <input.pt> <reference_jacobian.pt> <tolerance>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];
    std::string reference_jacobian_path = argv[3];
    double tolerance = std::atof(argv[4]);

    try {
        std::cout << "=== C++ Jacobian Validation Test ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Input: " << input_path << std::endl;
        std::cout << "Reference: " << reference_jacobian_path << std::endl;
        std::cout << "Tolerance: " << tolerance << std::endl;

        // Load model
        std::cout << "\n[1] Loading TorchScript model..." << std::endl;
        torch::jit::script::Module model = torch::jit::load(model_path);
        model.eval();
        std::cout << "    Model loaded successfully" << std::endl;

        // Load test input
        std::cout << "\n[2] Loading test input..." << std::endl;
        std::vector<char> input_buffer;
        std::ifstream input_file(input_path, std::ios::binary);
        if (!input_file) {
            throw std::runtime_error("Failed to open input file: " + input_path);
        }
        input_buffer.assign(std::istreambuf_iterator<char>(input_file),
                           std::istreambuf_iterator<char>());
        input_file.close();

        auto input_ivalue = torch::jit::pickle_load(input_buffer);
        torch::Tensor input = input_ivalue.toTensor();
        std::cout << "    Input shape: " << input.sizes() << std::endl;

        // Load reference Jacobian
        std::cout << "\n[3] Loading reference Jacobian (Python)..." << std::endl;
        std::vector<char> jac_buffer;
        std::ifstream jac_file(reference_jacobian_path, std::ios::binary);
        if (!jac_file) {
            throw std::runtime_error("Failed to open reference Jacobian file: " + reference_jacobian_path);
        }
        jac_buffer.assign(std::istreambuf_iterator<char>(jac_file),
                         std::istreambuf_iterator<char>());
        jac_file.close();

        auto jac_ivalue = torch::jit::pickle_load(jac_buffer);
        torch::Tensor reference_jacobian = jac_ivalue.toTensor();
        std::cout << "    Reference Jacobian shape: " << reference_jacobian.sizes() << std::endl;
        std::cout << "    Reference Jacobian mean: " << reference_jacobian.mean().item<double>() << std::endl;

        // Compute Jacobian using C++ jac_physical method
        std::cout << "\n[4] Computing Jacobian using C++ jac_physical()..." << std::endl;
        auto method = model.get_method("jac_physical");
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        torch::Tensor cpp_jacobian = method(inputs).toTensor();
        std::cout << "    C++ Jacobian shape: " << cpp_jacobian.sizes() << std::endl;
        std::cout << "    C++ Jacobian mean: " << cpp_jacobian.mean().item<double>() << std::endl;

        // Print first few Jacobian values for debugging
        std::cout << "\nC++ Jacobian values (first 5x5 block):" << std::endl;
        auto cpp_jac_flat = cpp_jacobian.flatten();
        int num_to_print = std::min(25, (int)cpp_jac_flat.size(0));
        for (int i = 0; i < num_to_print; ++i) {
            if (i % 5 == 0 && i > 0) {
                std::cout << std::endl;
            }
            std::cout << std::scientific << std::setprecision(6) << std::setw(13)
                      << cpp_jac_flat[i].item<double>() << " ";
        }
        std::cout << std::endl;

        std::cout << "\nPython Reference Jacobian values (first 5x5 block):" << std::endl;
        auto ref_jac_flat = reference_jacobian.flatten();
        num_to_print = std::min(25, (int)ref_jac_flat.size(0));
        for (int i = 0; i < num_to_print; ++i) {
            if (i % 5 == 0 && i > 0) {
                std::cout << std::endl;
            }
            std::cout << std::scientific << std::setprecision(6) << std::setw(13)
                      << ref_jac_flat[i].item<double>() << " ";
        }
        std::cout << std::endl;

        // Compare
        std::cout << "\n[5] Comparing C++ vs Python Jacobians..." << std::endl;
        ComparisonResult result = compare_tensors(cpp_jacobian, reference_jacobian, tolerance);
        print_result(result, tolerance);

        return result.passed ? 0 : 1;

    } catch (const c10::Error& e) {
        std::cerr << "Torch error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

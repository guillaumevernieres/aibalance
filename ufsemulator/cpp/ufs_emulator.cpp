#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <unordered_map>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.ts>\n";
    return 1;
  }
  std::string model_path = argv[1];

  // Load TorchScript model
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path);
  } catch (const c10::Error& e) {
    std::cerr << "Error loading model: " << e.what() << "\n";
    return 1;
  }

  // Read serialized metadata: input_names, output_names, levels, meta
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<int> input_levels;
  std::vector<int> output_levels;
  try {
    // input_names
    auto input_iv = module.attr("input_names");
    for (const auto& v : input_iv.toListRef()) {
      input_names.push_back(v.toString()->string());
    }

    // input_levels
    auto input_lv = module.attr("input_levels");
    for (const auto& v : input_lv.toListRef()) {
      input_levels.push_back(v.toInt());
    }

    std::cout << "Input variables (" << input_names.size() << "):\n";
    for (size_t i = 0; i < input_names.size(); ++i) {
      std::cout << "  [" << i << "] " << input_names[i];
      if (i < input_levels.size() && input_levels[i] >= 0) {
        std::cout << " (level: " << input_levels[i] << ")";
      }
      std::cout << "\n";
    }

    // output_names
    auto output_iv = module.attr("output_names");
    for (const auto& v : output_iv.toListRef()) {
      output_names.push_back(v.toString()->string());
    }

    // output_levels
    auto output_lv = module.attr("output_levels");
    for (const auto& v : output_lv.toListRef()) {
      output_levels.push_back(v.toInt());
    }

    std::cout << "Output variables (" << output_names.size() << "):\n";
    for (size_t i = 0; i < output_names.size(); ++i) {
      std::cout << "  [" << i << "] " << output_names[i];
      if (i < output_levels.size() && output_levels[i] >= 0) {
        std::cout << " (level: " << output_levels[i] << ")";
      }
      std::cout << "\n";
    }

    // meta dict
    std::unordered_map<std::string, std::string> meta;
    auto meta_iv = module.attr("meta");
    for (const auto& kv : meta_iv.toGenericDict()) {
      meta[kv.key().toString()->string()] = kv.value().toString()->string();
    }
    std::cout << "\nMeta entries: " << meta.size() << "\n";
    for (const auto& kv : meta) {
      std::cout << "  " << kv.first << ": " << kv.second << "\n";
    }

    // Summary: Group variables by level
    std::cout << "\n=== Level Summary ===\n";
    std::unordered_map<int, std::vector<std::string>> inputs_by_level;
    std::unordered_map<int, std::vector<std::string>> outputs_by_level;

    for (size_t i = 0; i < input_names.size(); ++i) {
      inputs_by_level[input_levels[i]].push_back(input_names[i]);
    }
    for (size_t i = 0; i < output_names.size(); ++i) {
      outputs_by_level[output_levels[i]].push_back(output_names[i]);
    }

    std::cout << "Input variables by level:\n";
    for (const auto& kv : inputs_by_level) {
      if (kv.first >= 0) {
        std::cout << "  Level " << kv.first << ": " << kv.second.size() << " variable(s)\n";
      } else {
        std::cout << "  No level specified: " << kv.second.size() << " variable(s)\n";
      }
    }

    std::cout << "Output variables by level:\n";
    for (const auto& kv : outputs_by_level) {
      if (kv.first >= 0) {
        std::cout << "  Level " << kv.first << ": " << kv.second.size() << " variable(s)\n";
      } else {
        std::cout << "  No level specified: " << kv.second.size() << " variable(s)\n";
      }
    }
  } catch (const c10::Error& e) {
    std::cerr << "Warning: metadata not found or invalid: " << e.what() << "\n";
  }

  // Derive input_size from input_names if available, else infer from first linear layer
  int64_t input_size = -1;
  if (!input_names.empty()) {
    input_size = static_cast<int64_t>(input_names.size());
  } else {
    // Try to infer from parameters: first weight of first Linear should be [hidden_size, input_size]
    for (const auto& p : module.named_parameters()) {
      if (p.name.find("0.weight") != std::string::npos && p.value.sizes().size() == 2) {
        input_size = p.value.size(1);
        break;
      }
    }
    if (input_size < 0) {
      std::cerr << "Unable to determine input size from model.\n";
      return 1;
    }
  }
  std::cout << "Inferred input_size: " << input_size << "\n";

  // Device
  torch::Device device(torch::kCPU);
  module.to(device);
  module.eval();

  // Example input: [1, input_size]
  torch::Tensor x = torch::randn({1, input_size}, torch::dtype(torch::kFloat)).to(device);

  // Inference
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(x);
  torch::Tensor y = module.forward(inputs).toTensor();
  std::cout << "Output: " << y << std::endl;

  // Jacobian dy/dx: [output_size, input_size]
  torch::Tensor x_req = x.detach().clone().set_requires_grad(true);
  std::vector<torch::jit::IValue> inputs2;
  inputs2.push_back(x_req);
  torch::Tensor y2 = module.forward(inputs2).toTensor(); // [1, output_size]
  int64_t output_size = y2.size(1);

  std::vector<torch::Tensor> jac_rows;
  for (int64_t i = 0; i < output_size; ++i) {
    if (x_req.grad().defined()) x_req.grad().zero_();
    torch::Tensor grad_out = torch::zeros_like(y2);
    grad_out.index_put_({0, i}, 1.0);
    y2.backward(grad_out, /*keep_graph=*/true);
    if (!x_req.grad().defined()) {
      std::cerr << "Gradient not computed\n";
      return 1;
    }
    jac_rows.push_back(x_req.grad().detach().clone()); // [1, input_size]
  }
  torch::Tensor jac = torch::stack(jac_rows).squeeze(1); // [output_size, input_size]
  std::cout << "Jacobian:\n" << jac << std::endl;

  return 0;
}

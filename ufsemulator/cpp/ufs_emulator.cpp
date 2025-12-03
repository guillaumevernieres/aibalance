#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model.ts> <input_size>\n";
    return 1;
  }
  std::string model_path = argv[1];
  int64_t input_size = std::stoll(argv[2]);

  // Load TorchScript model
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path);
  } catch (const c10::Error& e) {
    std::cerr << "Error loading model: " << e.what() << "\n";
    return 1;
  }

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

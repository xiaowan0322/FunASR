#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/serialize.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <memory>


float run(int i, torch::jit::script::Module &module,
          std::vector<torch::jit::IValue> &inputs) {
  torch::cuda::synchronize();
  auto start = std::chrono::system_clock::now();
  module.forward(inputs).toTuple();
  torch::cuda::synchronize();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << i << " " << 1000.0 * elapsed_seconds.count() << std::endl;
  return 1000.0 * elapsed_seconds.count();
}

std::vector<torch::IValue> load_data(const char* data_file) {
  std::ifstream file(data_file, std::ios::binary);
  std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  torch::IValue ivalue = torch::pickle_load(data);
  CHECK(ivalue.isTuple());
  torch::Tensor speech = ivalue.toTuple()->elements()[0].toTensor();
  torch::Tensor length = ivalue.toTuple()->elements()[1].toTensor();
  std::vector<torch::IValue> inputs{speech, length};
  return inputs;
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module> \n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
    torch::NoGradGuard no_grad;
    module.eval();
    for (int i = 0; i < 7176; ++ i) {
	char ch[50];
	sprintf(ch, "test_data/test_%d.pth", i);
    	auto inputs = load_data(ch);
	run(i, module, inputs);
    }
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model" << std::endl << e.what();
    return -1;
  }

  std::cout << "ok\n";
}

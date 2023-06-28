#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/serialize.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <memory>

int benchmark(torch::jit::script::Module &module,
             std::vector<torch::jit::IValue> &inputs) {
  // warmup 20-iter
  for (int k = 0; k < 20; ++ k) {
    module.forward(inputs);
    torch::cuda::synchronize();
  }
  auto start = std::chrono::system_clock::now();
  // run 100-iter
  for (int k = 0; k < 100; ++ k) {
    module.forward(inputs);
    torch::cuda::synchronize();
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
            << "\nelapsed time: " << elapsed_seconds.count() << "s"
            << "\navg latency: " << 1000.0 * elapsed_seconds.count()/100 << "ms\n";
  return 0;
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
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module> <path-to-saved-test-data>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    auto inputs = load_data(argv[2]);
    torch::NoGradGuard no_grad;
    module.eval();

    benchmark(module, inputs);
    auto outputs = module.forward(inputs);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model" << std::endl << e.what();
    return -1;
  }

  std::cout << "ok\n";
}

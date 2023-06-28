#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/serialize.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <memory>


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

std::vector<std::vector<torch::IValue>> load_data_lst(int start_i, int end_i) {
    std::vector<std::vector<torch::IValue>> data_lst;
    for (int i = start_i; i < end_i; ++ i) {
        char ch[50];
        sprintf(ch, "test_data/test_%d.pth", i);
        auto inputs = load_data(ch);
        data_lst.push_back(inputs);
    }
    return data_lst;
}

float infer(torch::jit::script::Module &module,
            std::vector<std::vector<torch::IValue>> data_lst,
            int warmup_num=0) {
    // std::vector<float> val(80, 560);
    torch::cuda::synchronize();
    int num = warmup_num == 0 ? data_lst.size() : warmup_num;
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < num; ++ i) {
        // torch::Tensor feats = torch::zeros({1, 80, 560}, torch::kFloat);
        // torch::Tensor feats = torch::from_blob(
        //     const_cast<float*>(val.data()), {1, 80, 560}, torch::kFloat);
        module.forward(data_lst[i]).toTuple();
        torch::cuda::synchronize();
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return 1000.0 * elapsed_seconds.count();
}

int main(int argc, const char* argv[]) {
    if (argc != 4) {
      std::cerr << "usage: example-app <path-to-exported-script-module> <start-id> <end-id>\n";
      return -1;
    }

    torch::jit::script::Module module = torch::jit::load(argv[1]);
    torch::NoGradGuard no_grad;
    module.eval();

    int start_i = atoi(argv[2]);
    int end_i = atoi(argv[3]);
    auto data_lst = load_data_lst(start_i, end_i);

    float t0 = infer(module, data_lst, 50);
    std::cout << "t0: " << t0 << std::endl;
    float t1 = infer(module, data_lst);
    std::cout << "t1: " << t1 << std::endl;
    float t2 = infer(module, data_lst);
    std::cout << "t2: " << t2 << std::endl;
}

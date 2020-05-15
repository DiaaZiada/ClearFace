#include "module.h"


Module::Module(std::string path){
    module = torch::jit::load(path);
}

torch::Tensor Module::forward(std::vector<torch::jit::IValue> inputs)
{
    torch::Tensor outputs = module.forward(inputs).toTensor(); 
    return outputs.clone();
}
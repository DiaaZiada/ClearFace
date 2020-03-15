#include "module/module.h"

Module* gender_module;

void load_module(std::string path){
  gender_module = new Module(path);
}

torch::Tensor forward(torch::Tensor inputs){
  std::vector<torch::jit::IValue> vector_inputs;
  vector_inputs.push_back(inputs);
  torch::Tensor output = gender_module->forward(std::move(vector_inputs));
  return output ;
}


static auto registry =
  torch::RegisterOperators("gender::load_module", &load_module).op("gender::forward", &forward);
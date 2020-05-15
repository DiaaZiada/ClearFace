#include "module/module.h"

Module* expression_module;

void load_module(std::string path){
  expression_module = new Module(path);
}

torch::Tensor forward(torch::Tensor inputs){
  std::vector<torch::jit::IValue> vector_inputs;
  vector_inputs.push_back(inputs);
  torch::Tensor output = expression_module->forward(std::move(vector_inputs));
  return output ;
}


static auto registry =
  torch::RegisterOperators("expression::load_module", &load_module).op("expression::forward", &forward);
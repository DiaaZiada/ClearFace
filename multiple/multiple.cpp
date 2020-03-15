#include "module/module.h"
#include <torch/torch.h>

std::string pathAppend(std::string p1, std::string p2) {

   char sep = '/';
   std::string tmp = p1;

#ifdef _WIN32
  sep = '\\';
#endif

  if (p1[p1.length()-1] != sep) {
     tmp += sep;                
     return(tmp + p2);
  }
  else
     return(p1 + p2);
}

Module *feature_extractor, *age, *makeup, *pose, *illumination, *occlusion;

void load_module(std::string path){
  std::string path_feature_extractor = pathAppend(path, "feature_extractor.zip");
  std::string path_age = pathAppend(path, "age.zip");
  std::string path_makeup = pathAppend(path, "makeup.zip");
  std::string path_pose = pathAppend(path, "pose.zip");
  std::string path_illumination = pathAppend(path, "illumination.zip");
  std::string path_occlusion = pathAppend(path, "occlusion.zip");
  
  feature_extractor = new Module(path_feature_extractor);
  age = new Module(path_age);
  makeup = new Module(path_makeup);
  pose = new Module(path_pose);
  illumination = new Module(path_illumination);
  occlusion = new Module(path_occlusion);

}

torch::Tensor forward_feature_extractor(torch::Tensor inputs){
  std::vector<torch::jit::IValue> vector_inputs;
  vector_inputs.push_back(inputs);
  torch::Tensor output = feature_extractor->forward(std::move(vector_inputs));
  return output;
}

torch::Tensor forward_age(torch::Tensor inputs){
  std::vector<torch::jit::IValue> vector_inputs;
  vector_inputs.push_back(inputs);
  torch::Tensor output = age->forward(std::move(vector_inputs));
  return output;
}

torch::Tensor forward_makeup(torch::Tensor inputs){
  std::vector<torch::jit::IValue> vector_inputs;
  vector_inputs.push_back(inputs);
  torch::Tensor output = makeup->forward(std::move(vector_inputs));
  return output;
}

torch::Tensor forward_pose(torch::Tensor inputs){
  std::vector<torch::jit::IValue> vector_inputs;
  vector_inputs.push_back(inputs);
  torch::Tensor output = pose->forward(std::move(vector_inputs));
  return output;
}

torch::Tensor forward_illumination(torch::Tensor inputs){
  std::vector<torch::jit::IValue> vector_inputs;
  vector_inputs.push_back(inputs);
  torch::Tensor output = illumination->forward(std::move(vector_inputs));
  return output;
}


torch::Tensor forward_occlusion(torch::Tensor inputs){
  std::vector<torch::jit::IValue> vector_inputs;
  vector_inputs.push_back(inputs);
  torch::Tensor output = occlusion->forward(std::move(vector_inputs));
  return output;
}


static auto registry =
  torch::RegisterOperators("multiple::load_module", &load_module).
  op("multiple::feature_extractor", &forward_feature_extractor).
  op("multiple::age", &forward_age).
  op("multiple::makeup", &forward_makeup).
  op("multiple::pose", &forward_pose).
  op("multiple::illumination", &forward_illumination).
  op("multiple::occlusion", &forward_occlusion);

#include <iostream>
#include <torch/script.h>



class Module{

    private:
        torch::jit::script::Module module;

    public:
        Module(std::string);
        at::Tensor forward(std::vector<torch::jit::IValue>);

};
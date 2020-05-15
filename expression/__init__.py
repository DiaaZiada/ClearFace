import torch

torch.ops.load_library("./expression/build/libexpression.so")

class Expression:
    def __init__(self, path):
        torch.ops.expression.load_module(path)
    def __call__(self, inp):
        return torch.ops.expression.forward(inp)



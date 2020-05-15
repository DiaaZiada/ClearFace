import torch

torch.ops.load_library("./gender/build/libgender.so")

class Gender:
    def __init__(self, path):
        torch.ops.gender.load_module(path)
    def __call__(self, inp):
        return torch.ops.gender.forward(inp)



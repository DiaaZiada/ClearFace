import torch

torch.ops.load_library("./multiple/build/libmultiple.so")

class Multiple:
    def __init__(self, path):
        torch.ops.multiple.load_module(path)
    def __call__(self, inp):

        features = torch.ops.multiple.feature_extractor(inp)
        illumination = torch.ops.multiple.illumination(features)
        pose = torch.ops.multiple.pose(features)
        occlusion = torch.ops.multiple.occlusion(features)
        age = torch.ops.multiple.age(features)
        makeup = torch.ops.multiple.makeup(features)

        return [illumination, pose, occlusion, age, makeup]
        

        



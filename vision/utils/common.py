import torch

MEAN = torch.unsqueeze(torch.unsqueeze(torch.Tensor((0.485, 0.456, 0.406)), -1), -1)
STD = torch.unsqueeze(torch.unsqueeze(torch.Tensor((0.229, 0.224, 0.225)), -1), -1)

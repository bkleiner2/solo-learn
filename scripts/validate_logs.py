import sys
import torch

path = sys.argv[1]

test = torch.load(path)
print (test.min(), test.max(), test.shape, torch.where(test == 0, 1, 0).sum())

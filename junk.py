import torch

m = torch.nn.Linear(5, 4)
input = torch.randn(3, 5)
output = m(input)
print ("input:  ", input)
print ("output: ", output)
print (output.size())

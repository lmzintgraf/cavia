import torch
import torch.nn as nn
import torch.nn.functional as F

class NMLinear(nn.Module):
	def __init__(self, input_channels, output_channels):
		super(NMLinear, self).__init__()
		self.weight = nn.Parameter(torch.Tensor(input_channels, output_channels), requires_grad=True)

		self.ws = nn.Linear(input_channels, 8) # Hardcode 8 for now, TODO: Make this a parameter in constructor this
		self.wm = nn.Linear(8, 8)
		self.wn = nn.Linear(8, output_channels)

		self.bias = nn.Parameter(torch.Tensor(output_channels), requires_grad=True)


	def forward(self, input):
		y = F.ReLU(self.wm(input))
		y_prime = F.tanh(self.wn(y))

		z = self.ws(input)

		output = F.ReLu(z * y_prime)
		return output

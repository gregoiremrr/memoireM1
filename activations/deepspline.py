import torch
import torch.nn as nn

class DeepSpline(nn.Module):
    def __init__(self):
        super(DeepSpline, self).__init__()
        self.b = nn.Parameter(torch.tensor([1.], requires_grad=True))
        self.p1 = nn.Parameter(torch.tensor([1.], requires_grad=True))
        self.p2 = nn.Parameter(torch.tensor([1.], requires_grad=True))
        self.p3 = nn.Parameter(torch.tensor([1.], requires_grad=True))

        self.x1 = 0  # nn.Parameter(torch.tensor([1.], requires_grad=True))
        self.x2 = 1  # nn.Parameter(torch.tensor([1.], requires_grad=True))

    def forward(self, x):
        coeff1 = self.p1 / max(1, abs(self.p1))
        part1 = (coeff1 * x + self.b) * (x < self.x1)
        coeff2 = self.p2 / max(1, abs(self.p2))
        b2 = coeff1 * self.x1 + self.b
        part2 = (coeff2 * (x - self.x1) + b2) * (self.x1 <= x) * (x < self.x2)
        coeff3 = self.p3 / max(1, abs(self.p3))
        b3 = coeff2 * (self.x2 - self.x1) + b2
        part3 = (coeff3 * (x - self.x2) + b3) * (self.x2 <= x)
        return part1 + part2 + part3


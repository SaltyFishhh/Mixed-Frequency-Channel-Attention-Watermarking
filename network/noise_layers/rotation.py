import torch.nn as nn
from kornia.geometry import transform
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Rotation(nn.Module):
    def __init__(self, angle):
        super(Rotation, self).__init__()
        self.angle = torch.tensor([angle]).float().to(device)

    def forward(self, image_cover):
        image, cover_image = image_cover
        output = transform.rotate(image, self.angle)
        return output
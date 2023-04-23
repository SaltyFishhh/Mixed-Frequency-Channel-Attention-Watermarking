from kornia.geometry import transform
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Scale(nn.Module):
    def __init__(self, x_factor, y_factor):
        super(Scale, self).__init__()
        self.scale_factor = torch.tensor([[x_factor, y_factor]]).float().to(device)

    def forward(self, image_cover):
        image, cover_image = image_cover
        output = transform.scale(image, self.scale_factor)
        return output
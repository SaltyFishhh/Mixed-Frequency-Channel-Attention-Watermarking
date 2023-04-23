from kornia.enhance.adjust import sharpness
import torch.nn as nn

class Sharpen(nn.Module):

    def __init__(self, factor):
        super(Sharpen, self).__init__()
        self.factor = factor

    def forward(self, input):
        return sharpness(input, self.factor)

class SP(nn.Module):

    def __init__(self, factor):
        super(SP, self).__init__()
        self.sharpness = Sharpen(factor)

    def forward(self, image_cover):
        image, cover_image = image_cover
        return self.sharpness(image)
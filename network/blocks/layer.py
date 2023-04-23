import torch.nn as nn
import torch
import math
from torch.nn import functional as F
"""create a high frequence channel attention module"""

def get_freq_indicecs(method):
    # assert断言类似于if，后面的表达式成立，可以继续运行
    assert method in ['high1', 'high2', 'high4', 'high8', 'high16', 'high32',
                      'midd1', 'midd2', 'midd4', 'midd8', 'midd16', 'midd32',
                      'lowf1', 'lowf2', 'lowf4', 'lowf8', 'lowf16', 'lowf32']
    # get the last number of method
    num_freq = int(method[4:])
    if 'high' in method:
        # select the high frequency parts as indices
        all_high_indices_x = [7, 7, 6, 6, 5, 7, 5, 6, 5, 4, 7, 3, 7, 4, 6, 2, 7, 3, 6, 4, 5, 1, 7, 6, 2, 5, 3, 4, 0, 7, 1, 6]
        all_high_indices_y = [7, 6, 7, 6, 7, 5, 6, 5, 5, 7, 4, 7, 3, 6, 4, 7, 2, 6, 3, 5, 4, 7, 1, 2, 6, 3, 5, 4, 7, 0, 6, 1]
        mapper_x = all_high_indices_x[:num_freq]
        mapper_y = all_high_indices_y[:num_freq]
    elif 'midd' in method:
        # zig-zag 形选择频率分量，一般选择16个频率分量
        all_middle_indices_x = [3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 5, 4]
        all_middle_indices_y = [3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 4, 3, 2, 1, 0, 0, 1, 2]
        mapper_x = all_middle_indices_x[:num_freq]
        mapper_y = all_middle_indices_y[:num_freq]
    elif 'lowf' in method:
        all_low_indices_x = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3]
        all_low_indices_y = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralDCTlayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTlayer, self).__init__()
        assert len(mapper_x) == len(mapper_y)
        # the number of channel must be a multiple of the length and width
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        # self.register_buffer() 保存模型的参数，但是它是固定的，每次optimizer.step，不会更新参数
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weoght', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got' + str(len(x.shape))
        # n,c,h,w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        """
        Get DCT filters
        :param tile_size_x: the width of the input
        :param tile_size_y: the height of the input
        :param mapper_x: the number of the divided block(indices)
        :param mapper_y: the number of the divided block(indices)
        :param channel:
        :return: Designed dct filters
        """
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        # enumerate([]) 遍历列表中的element，返回该element和its indices
        # zip([iterable,...]),eg mapper_x and mapper_y 是list,把两个list中的element打包成tuple的list
        # a = [1, 2, 3],b = [4, 5, 6],zip(a,b) = [(1,4), (2,5), (3,6)]
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter

class MultiSpectralAttention(nn.Module):
    """
    Create a Multi-spectral Attention Block
    """
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'high16'):
        super(MultiSpectralAttention, self).__init__()
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indicecs(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 8) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 8) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 8x8 frequency space
        # eg, (2,2) in 16x16 is identical to (1,1) in 8x8

        self.dct_layer = MultiSpectralDCTlayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        # pass input image through dct_filter, get dct feature map
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        # y.expand_as(tensor) 把张量y扩展为参数tensor的大小
        return x * y.expand_as(x)
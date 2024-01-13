import torch
import torch.nn as nn
import numpy as np


class Pos_Embed_Sinusoid(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(Pos_Embed_Sinusoid, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):

            # 生成 1, dim 的行向量
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # 将 n_position 个 1 x dim。也就是 n_position, dim。
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

        # # 从 0 列开始，每隔 2 取值
        # n_position, dim/2。
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i

        # 从 1 列开始，每隔 2 取值
        # n_position, dim/2
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        # 1, n_position, dim
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        
        B, C, H, W = (x.size())

        # 这里是取 pos_embed 与 x 的等长部分
        # 1, n_position, dim
        pos_embed = self.pos_table[:, :H*W].clone().detach()

        # 1, n_position, dim ---> 1, H, W, C ---> 1, C, H, W
        pos_embed_reshaped = pos_embed.reshape(1, H, W, C).permute(0, 3, 1, 2)

        add_pos_embed = x + pos_embed_reshaped

        return add_pos_embed

        # # 1, n_position, dim
        # pos_embed = self.pos_table[:, :HW].clone().detach()

        # return pos_embed
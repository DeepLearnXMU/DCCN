import sys
sys.setrecursionlimit(15000)
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class CoAttention(nn.Module):
    def __init__(self):
        super(CoAttention, self).__init__()

    def forward(self, img, text, img_mask, text_mask, num_regions):
        """
        PCC.
        Args:
            img: visual features.
            text: text context C[batch, len, num_cap, dim].
            img_mask: object mask for regional visual features.
            text_mask: text mask.
            num_regions: max object number.
        """
        img = img.unsqueeze(1).expand(-1, text.shape[1], -1, -1, -1)
        text = text.unsqueeze(2).expand(-1, -1, num_regions, -1, -1)
        vimg = img - torch.mean(img)
        vtext = text - torch.mean(text)
        weight_before_tanh = torch.sum(vimg * vtext, dim=-1) / (torch.sqrt(torch.sum(vimg ** 2)) * torch.sqrt(torch.sum(vtext ** 2)))
        if text_mask is not None:
            text_mask = text_mask.unsqueeze(-1).unsqueeze(-1)
            weight_before_tanh = weight_before_tanh.masked_fill(Variable(text_mask), 0)
        if img_mask is not None:
            img_mask = img_mask.unsqueeze(1).unsqueeze(-1)
            weight_before_tanh = weight_before_tanh.masked_fill(img_mask, 0)
        weight = F.tanh(weight_before_tanh)
        return weight

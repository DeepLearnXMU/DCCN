import sys
sys.setrecursionlimit(15000)
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import io
from onmt.modules.CoAttention import CoAttention

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class CapsuleNet(nn.Module):
    """
    context-guided capsule network.
    Args:
        num_iterations: interation times N_{itr}.
        num_capsules: number of high-level capsules N_v.
        num_regions: number of low-level capsules N_u.
            (196 for global visual features and max objects number for regional visual features).
        dim: model dimension.
    """
    def __init__(self, num_iterations, num_capsules, num_regions, dim):
        super(CapsuleNet, self).__init__()
        self.dp = nn.Dropout(0.1)
        # W_{ij} that transforms low-level capsules to high-level capsules.
        self.route_weights = nn.Parameter(torch.Tensor(dim, num_capsules * dim))
        self.num_regions = num_regions
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        # correlation function (PCC)
        self.coatt = CoAttention()
        # "FuseMultimodalContext" in algorithm
        self.fusion_layer = nn.Linear(num_capsules*dim, dim)
        self.W_m = nn.Linear(dim, dim)
        self.W_output = nn.Linear(dim, dim)

    def forward(self, img, cxt, img_mask=None, text_mask=None):
        """
        routing algorithm.
        Args:
            img: visual features I,
                 can be global[batch, 196, num_cap, dim] or regional[batch, num_regions, num_cap, dim].
            cxt: text context C[batch, len, num_cap, dim].
            img_mask: object mask for regional visual features, [batch, num_regions].
            text_mask: [batch, len].
        """
        if self.num_regions == 196:
            img_mask = None
        if img_mask is not None:
            img_mask = img_mask.byte()
        u = img.unsqueeze(2).repeat(1, 1, self.num_capsules, 1)
        m = cxt.unsqueeze(2).repeat(1, 1, self.num_capsules, 1)
        weight = self.coatt(u, self.W_m(m), img_mask, text_mask, self.num_regions)
        b = Variable(torch.zeros(u.shape[0], cxt.shape[1], self.num_regions, self.num_capsules)).cuda()
        img = img.unsqueeze(1).repeat(1, cxt.shape[1], 1, 1)
        # prepare mask for text and objects
        if text_mask is not None:
            tmp_mask = text_mask
            tmp_mask = tmp_mask.unsqueeze(-1).unsqueeze(-1)
            b = b.masked_fill(Variable(tmp_mask), -1e18)
        if img_mask is not None:
            tmp_mask = img_mask
            tmp_mask = tmp_mask.unsqueeze(1).unsqueeze(-1)
            b = b.masked_fill(tmp_mask, -1e18)
        priors = (img @ self.route_weights).view(img.shape[0], img.shape[1], self.num_regions, self.num_capsules, -1)
        for i in range(self.num_iterations):
            c = F.softmax(b, dim=2)
            c = c + weight
            c = c.unsqueeze(-1)
            # generate higl-level capsules and multimodal context capsules
            outputs = (c * priors).sum(dim=2)
            m = self.dp(m * self.W_output(outputs))
            # update rho and b
            weight = self.coatt(u, self.W_m(m), img_mask, text_mask, self.num_regions)
            delta_b = (weight.unsqueeze(-1) * priors * outputs.unsqueeze(2)).sum(dim=-1)
            b = b + delta_b

        cxt = m.view(m.shape[0], m.shape[1], -1)
        cxt = self.fusion_layer(cxt)
        return outputs, cxt


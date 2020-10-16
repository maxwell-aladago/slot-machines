from math import sqrt

import torch.nn.functional as F
from selection_engine import GreedySelection, ProbabilisticSelection
from torch import nn, empty, sum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, k, greedy_selection=True):
        super(Linear, self).__init__()
        self.k = k

        # the weights retain their initial values through training
        self.weight = nn.Parameter(empty((out_features, in_features, self.k)), requires_grad=False)
        self.score = nn.Parameter(empty(out_features, in_features, self.k))

        self.xavier_uniform_()

        if greedy_selection:
            self._selection_engine = GreedySelection.apply
        else:
            self._selection_engine = ProbabilisticSelection.apply

    def forward(self, x):
        selected_net = self._selection_engine(self.score)
        net = sum(self.weight * selected_net, dim=-1)
        out = F.linear(x.view(x.size(0), -1), net)

        return out

    def xavier_uniform_(self):
        w, h, k = self.weight.size()
        std = sqrt(6.0 / float(w + h))
        self.weight.data.uniform_(-std, std)
        self.score.data.uniform_(0, std)

r'''
This is the same as neuralsort.py, but instead of being hardcoded into GPU it
allows using either GPU or CPU. (Compare 'forward' method to the on in neuralsort.py)
'''

import torch
from torch import Tensor


class NeuralSort(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False, device='cuda'):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.device = device
        if device == 'cuda':
            self.torch = torch.cuda
        elif device == 'cpu':
            self.torch = torch
        else:
            raise ValueError('Unknown device: %s' % device)

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        one = self.torch.FloatTensor(dim, 1).fill_(1)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        # B = torch.matmul(A_scores, torch.matmul(
        #     one, torch.transpose(one, 0, 1)))  # => NeuralSort O(n^3) BUG!
        B = torch.matmul(torch.matmul(A_scores,
                         one), torch.transpose(one, 0, 1))  # => Bugfix
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
                   ).type(self.torch.FloatTensor)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device=self.device)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(
                dim0=1, dim1=0).flatten().type(self.torch.LongTensor)
            r_idx = torch.arange(dim).repeat(
                [bsize, 1]).flatten().type(self.torch.LongTensor)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat

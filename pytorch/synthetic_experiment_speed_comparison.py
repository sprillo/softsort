import argparse
import time
from scipy import stats

import numpy as np
import torch
from torch.autograd import Variable

from neuralsort_cpu_or_gpu import NeuralSort
from softsort import SoftSort_p2

parser = argparse.ArgumentParser(description="Benchmark speed of softsort vs"
                                 " neuralsort")

parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--n", type=int, default=2000)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--method", type=str, default='neuralsort')
parser.add_argument("--burnin", type=int, default=100)

args = parser.parse_args()

print("Benchmarking with:\n"
      "\tbatch_size = %d\n"
      "\tn = %d\n"
      "\tepochs = %d\n"
      "\tdevice = %s\n"
      "\tmethod = %s\n"
      "\tburnin = %d" %
      (args.batch_size,
       args.n,
       args.epochs,
       args.device,
       args.method,
       args.burnin))

np.random.seed(1)
torch.manual_seed(1)

sort_op = None
if args.method == 'neuralsort':
    sort_op = NeuralSort(tau=100.0, device=args.device)
elif args.method == 'softsort':
    sort_op = SoftSort_p2(tau=0.1)
else:
    raise ValueError('method %s not found' % args.method)

scores = Variable(torch.rand(size=(args.batch_size, args.n),
                  device=args.device) * 2.0 - 1.0, requires_grad=True)
optimizer = torch.optim.SGD([scores], lr=10.0, momentum=0.5, weight_decay=0.01)


def evaluate(scores):
    r'''
    Returns the mean spearman correlation over the batch.
    '''
    scores_eval = scores.cpu().detach().numpy()
    rank_correlations = []
    for i in range(args.batch_size):
        rank_correlation, _ = stats.spearmanr(scores_eval[i], range(args.n, 0, -1))
        rank_correlations.append(rank_correlation)
    mean_rank_correlation = np.mean(rank_correlations)
    return mean_rank_correlation


def training_step():
    optimizer.zero_grad()

    # Normalize scores before feeding them into the sorting op for increased stability.
    min_scores, _ = torch.min(scores, dim=1, keepdim=True)
    min_scores = min_scores.detach()
    max_scores, _ = torch.max(scores, dim=1, keepdim=True)
    max_scores = max_scores.detach()
    scores_normalized = (scores - min_scores) / (max_scores - min_scores)
    P_hat = sort_op(scores_normalized)

    loss = torch.mean(1.0 - torch.log(torch.diagonal(P_hat, dim1=1, dim2=2)))
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss


# burn-in
for epoch in range(args.burnin):
    training_step()
if args.device == 'cuda':
    torch.cuda.synchronize()

# train
start_time = time.time()
log = ""
for epoch in range(args.epochs):
    loss = training_step()
spearmanr = evaluate(scores)
if args.device == 'cuda':
    torch.cuda.synchronize()
end_time = time.time()
total_time = end_time - start_time

log += "Epochs: %d\n" % args.epochs
log += "Loss: %f\n" % loss
log += "Spearmanr: %f\n" % spearmanr
log += "Total time: %f\n" % total_time
log += "Time per epoch: %f\n" % (total_time / args.epochs)

print(log)

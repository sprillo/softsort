import argparse
import os
import time

import numpy as np
import tensorflow as tf
from scipy import stats

import util

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

sort_op = None
if args.method == 'neuralsort':
    sort_op = util.neuralsort
    args.tau = 100.0
elif args.method == 'softsort':
    sort_op = util.softsort_p2
    args.tau = 0.03
else:
    raise ValueError('method %s not found' % args.method)

device_str = '/GPU:0' if args.device == 'cuda' else '/CPU:0'


def evaluate(scores_eval):
    r'''
    Returns the mean spearman correlation over the batch.
    '''
    rank_correlations = []
    for i in range(args.batch_size):
        rank_correlation, _ = stats.spearmanr(scores_eval[i, :, 0],
                                              range(args.n, 0, -1))
        rank_correlations.append(rank_correlation)
    mean_rank_correlation = np.mean(rank_correlations)
    return mean_rank_correlation


log = ""
with tf.Session() as sess:
    with tf.device(device_str):
        np.random.seed(1)
        tf.set_random_seed(1)
        # Define model
        scores = tf.get_variable(
            shape=[args.batch_size, args.n, 1],
            initializer=tf.random_uniform_initializer(-1.0, 1.0),
            name='scores')

        # Normalize scores before feeding them into the sorting op for increased stability.
        min_scores = tf.math.reduce_min(scores, axis=1, keepdims=True)
        min_scores = tf.stop_gradient(min_scores)
        max_scores = tf.math.reduce_max(scores, axis=1, keepdims=True)
        max_scores = tf.stop_gradient(max_scores)
        scores_normalized = (scores - min_scores) / (max_scores - min_scores)

        P_hat = sort_op(scores_normalized, tau=args.tau)

        wd = 5.0
        loss = (tf.reduce_mean(1.0 - tf.log(tf.matrix_diag_part(P_hat)))
                + wd * tf.reduce_mean(tf.multiply(scores, scores))) * args.batch_size
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=10.0,
            momentum=0.5).\
            minimize(loss, var_list=[scores])
    # Train model
    tf.global_variables_initializer().run()
    # Burn-in
    for _ in range(args.burnin):
        sess.run(optimizer)
    # Train
    start_time = time.time()
    for epoch in range(args.epochs):
        sess.run(optimizer)
    loss_eval, scores_eval = sess.run([loss, scores])
    spearmanr = evaluate(scores_eval)
    end_time = time.time()
    total_time = end_time - start_time
    log += "Epochs: %d\n" % args.epochs
    log += "Loss: %f\n" % loss_eval
    log += "Spearmanr: %f\n" % spearmanr
    log += "Total time: %f\n" % total_time
    log += "Time per epoch: %f\n" % (total_time / args.epochs)

print(log)

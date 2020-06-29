import numpy as np
import matplotlib.pyplot as plt

from run_sort_table_of_results import RunSortResultsParser


def get_filename(n, method, tau, lr, num_epochs, repetition):
    filename = "./run_sort_results/N_%s_%s/N_%s_%s_TAU_%s_LR_%s_E_%s_REP_%s.txt" %\
        (n, method, n, method, tau, lr, num_epochs, repetition)
    return filename


def get_learning_curves(n, method, tau, lr, num_epochs, repetitions):
    average_losses = np.zeros(shape=(num_epochs, len(repetitions)))
    val_set_prop_all_corrects = np.zeros(shape=(num_epochs, len(repetitions)))
    val_set_prop_any_corrects = np.zeros(shape=(num_epochs, len(repetitions)))
    for r_id, repetition in enumerate(repetitions):
        filename = get_filename(n, method, tau, lr, num_epochs, repetition)
        parser = RunSortResultsParser()
        parser.parse(filename, expected_length=num_epochs)
        for i in range(num_epochs):
            average_losses[i, r_id] = float(parser.average_loss[i])
            val_set_prop_all_corrects[i, r_id] = float(parser.val_set_prop_all_correct[i])
            val_set_prop_any_corrects[i, r_id] = float(parser.val_set_prop_any_correct[i])

    average_losses_mean = average_losses.mean(axis=1)
    average_losses_std = average_losses.std(axis=1)

    val_set_prop_all_corrects_mean = val_set_prop_all_corrects.mean(axis=1)
    val_set_prop_all_corrects_std = val_set_prop_all_corrects.std(axis=1)

    val_set_prop_any_corrects_mean = val_set_prop_any_corrects.mean(axis=1)
    val_set_prop_any_corrects_std = val_set_prop_any_corrects.std(axis=1)

    return average_losses_mean, average_losses_std, val_set_prop_all_corrects_mean, val_set_prop_all_corrects_std,\
        val_set_prop_any_corrects_mean, val_set_prop_any_corrects_std


ns = ['3', '5', '7', '9', '15']
taus = ['1024', '1024', '1024', '128', '128']
lr = '0.005'
num_epochs = 100
repetitions = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


for n, tau in zip(ns, taus):
    if n != '15':
        continue
    print(f"n = {n}")

    for det_or_stoch in ["deterministic", "stochastic"]:
        if det_or_stoch != "deterministic":
            continue
        average_losses_mean = {}
        average_losses_std = {}
        val_set_prop_all_corrects_mean = {}
        val_set_prop_all_corrects_std = {}
        val_set_prop_any_corrects_mean = {}
        val_set_prop_any_corrects_std = {}

        for method in [det_or_stoch + "_softsort", det_or_stoch + "_neuralsort"]:
            average_losses_mean[method],\
                average_losses_std[method],\
                val_set_prop_all_corrects_mean[method],\
                val_set_prop_all_corrects_std[method],\
                val_set_prop_any_corrects_mean[method],\
                val_set_prop_any_corrects_std[method] = get_learning_curves(n, method, tau, lr, num_epochs, repetitions)

        title_prefix = f"N = {n}\n"
        title_suffix = "\nAverage training curve over 10 repetitions"

        plt.figure(figsize=(7, 5))
        plt.plot(average_losses_mean[det_or_stoch + "_neuralsort"], label="NeuralSort", color='red', linestyle='--')
        plt.plot(average_losses_mean[det_or_stoch + "_softsort"], label="SoftSort", color='blue', linestyle='-')
        fontsize = 17
        plt.ylabel('Loss', fontsize=fontsize)
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig("../images/run_sort_learning_curve.png")

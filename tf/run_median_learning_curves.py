import numpy as np
import matplotlib.pyplot as plt

from run_median_table_of_results import RunMedianResultsParser


def get_filename(n, method, tau, lr, num_epochs, repetition):
    filename = "./run_median_results/N_%s_%s/N_%s_%s_TAU_%s_LR_%s_E_%s_REP_%s.txt" %\
        (n, method, n, method, tau, lr, num_epochs, repetition)
    return filename


def get_learning_curves(n, method, tau, lr, num_epochs, repetitions):
    average_losses = np.zeros(shape=(num_epochs, len(repetitions)))
    val_set_correctly_identifieds = np.zeros(shape=(num_epochs, len(repetitions)))
    val_set_mean_squared_errors = np.zeros(shape=(num_epochs, len(repetitions)))
    val_set_r2s = np.zeros(shape=(num_epochs, len(repetitions)))
    val_set_spearmanrs = np.zeros(shape=(num_epochs, len(repetitions)))
    for r_id, repetition in enumerate(repetitions):
        filename = get_filename(n, method, tau, lr, num_epochs, repetition)
        parser = RunMedianResultsParser()
        parser.parse(filename, expected_length=num_epochs)
        for i in range(num_epochs):
            average_losses[i, r_id] = float(parser.average_loss[i])
            val_set_correctly_identifieds[i, r_id] = float(parser.val_set_correctly_identified[i])
            val_set_mean_squared_errors[i, r_id] = float(parser.val_set_mean_squared_error[i])
            val_set_r2s[i, r_id] = float(parser.val_set_r2[i])
            val_set_spearmanrs[i, r_id] = float(parser.val_set_spearmanr[i])

    average_losses_mean = average_losses.mean(axis=1)
    average_losses_std = average_losses.std(axis=1)

    val_set_correctly_identifieds_mean = val_set_correctly_identifieds.mean(axis=1)
    val_set_correctly_identifieds_std = val_set_correctly_identifieds.std(axis=1)

    val_set_mean_squared_errors_mean = val_set_mean_squared_errors.mean(axis=1)
    val_set_mean_squared_errors_std = val_set_mean_squared_errors.std(axis=1)

    val_set_r2s_mean = val_set_r2s.mean(axis=1)
    val_set_r2s_std = val_set_r2s.std(axis=1)

    val_set_spearmanrs_mean = val_set_spearmanrs.mean(axis=1)
    val_set_spearmanrs_std = val_set_spearmanrs.std(axis=1)

    return average_losses_mean, average_losses_std, val_set_correctly_identifieds_mean,\
        val_set_correctly_identifieds_std, val_set_mean_squared_errors_mean, val_set_mean_squared_errors_std,\
        val_set_r2s_mean, val_set_r2s_std, val_set_spearmanrs_mean, val_set_spearmanrs_std


ns = ['5', '9', '15']
lr = '0.001'
num_epochs = 100
repetitions = ['0']


def get_tau_for_n_and_method(n, method):
    ns = ['5', '9', '15']
    methods = ['deterministic_neuralsort', 'stochastic_neuralsort', 'deterministic_softsort', 'stochastic_softsort']
    taus = [
        ['1024', '2048', '2048', '4096'],
        ['512', '512', '2048', '2048'],
        ['1024', '4096', '256', '2048']]

    for col, method2 in enumerate(methods):
        if method2 == method:
            for row, n2 in enumerate(ns):
                if n2 == n:
                    return taus[row][col]


for n in ns:
    if n != '15':
        continue
    for det_or_stoch in ["deterministic", "stochastic"]:
        if det_or_stoch != "deterministic":
            continue
        average_losses_mean = {}
        average_losses_std = {}

        val_set_correctly_identifieds_mean = {}
        val_set_correctly_identifieds_std = {}

        val_set_mean_squared_errors_mean = {}
        val_set_mean_squared_errors_std = {}

        val_set_r2s_mean = {}
        val_set_r2s_std = {}

        val_set_spearmanrs_mean = {}
        val_set_spearmanrs_std = {}

        for method in [det_or_stoch + "_softsort", det_or_stoch + "_neuralsort"]:
            tau = get_tau_for_n_and_method(n, method)
            average_losses_mean[method],\
                average_losses_std[method],\
                val_set_correctly_identifieds_mean[method],\
                val_set_correctly_identifieds_std[method],\
                val_set_mean_squared_errors_mean[method],\
                val_set_mean_squared_errors_std[method],\
                val_set_r2s_mean[method],\
                val_set_r2s_std[method],\
                val_set_spearmanrs_mean[method],\
                val_set_spearmanrs_std[method] = get_learning_curves(n, method, tau, lr, num_epochs, repetitions)

        title_prefix = f"N = {n}\n"
        title_suffix = "\nTraining curve"

        plt.figure(figsize=(7, 5))
        plt.plot(1e-6 * average_losses_mean[det_or_stoch + "_neuralsort"], label="NeuralSort", color='red',
                 linestyle='--')
        plt.plot(1e-6 * average_losses_mean[det_or_stoch + "_softsort"], label="SoftSort", color='blue',
                 linestyle='-')
        fontsize = 17
        plt.ylabel(r'Loss ($\times 10^{-6}$)', fontsize=fontsize)
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig("../images/run_median_learning_curve.png")

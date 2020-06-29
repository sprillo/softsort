import matplotlib.pyplot as plt


N = 4000
BATCH_SIZE = 20
ROOT_DIR = "benchmark_results_learning_curve"


def get_spearmanr2s_from_file(filename):
    spearmanr2s = []
    for line in open(filename):
        if line.startswith('Epoch '):
            spearmanr2s.append(float(line.split(' ')[-1]))
    return spearmanr2s


plt.figure(figsize=(7, 5))
fontsize = 16
spearmanr2s_neuralsort =\
    get_spearmanr2s_from_file(f"{ROOT_DIR}/benchmark_results_learning_curve_neuralsort_tf_{N}_{BATCH_SIZE}.txt")
plt.plot(spearmanr2s_neuralsort, label='NeuralSort', color='red', linestyle='--')
spearmanr2s_softsort_p1 =\
    get_spearmanr2s_from_file(f"{ROOT_DIR}/benchmark_results_learning_curve_softsort_p1_tf_{N}_{BATCH_SIZE}.txt")
plt.plot(spearmanr2s_softsort_p1, label='SoftSort, p=1', color='blue', linestyle='-.')
spearmanr2s_softsort_p2 =\
    get_spearmanr2s_from_file(f"{ROOT_DIR}/benchmark_results_learning_curve_softsort_p2_tf_{N}_{BATCH_SIZE}.txt")
plt.plot(spearmanr2s_softsort_p2, label='SoftSort, p=2', color='blue', linestyle='-')
# plt.title('Learning Curves for Benchmark Task')
plt.xlabel('Epoch', fontsize=fontsize)
plt.ylabel('Spearman R2', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig("images/synthetic_experiment_learning_curves.png")

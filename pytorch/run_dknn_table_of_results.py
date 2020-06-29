# Selects best runs according to validation accuracy
import glob
import os


def get_last_accuracy(file):
    with open(file) as f:
        lines = f.readlines()
        if len(lines) < 2:
            return (0, 0)
        val_acc_text, test_acc_text = lines[-2:]
        if 'val acc: ' not in val_acc_text:
            return (0, 0)
        val_acc = float(val_acc_text.split('val acc: ')[1])
        if 'test acc: ' not in test_acc_text:
            return (0, 0)
        test_acc = float(test_acc_text.split('test acc: ')[1])
        return val_acc, test_acc, os.path.basename(file)


def get_best_run(dir):
    return list(sorted(
        get_last_accuracy(file) for file in glob.glob(os.path.join(dir, '*'))
    ))[-1][1:]


for dir in glob.glob(os.path.join('run_dknn_results', '*')):
    print(dir, *get_best_run(dir))

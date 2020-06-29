import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class BenchmarkSoftsortBackwardsResultsParser:
    r'''
    Parses an individual results (i.e. log) file and stores the results.
    '''
    def __init__(self):
        self.epochs = None
        self.loss = None
        self.spearmanr = None
        self.total_time = None
        self.time_per_epoch = None
        self.oom = False

    def parse(self, file_path, expected_length=None):
        r'''
        :param file_path: path to the results (i.e. log) file
        '''
        with open(file_path) as file:
            for line in file:
                line_tokens = line.replace(',', '').replace('\n', '').split(' ')
                if line.startswith("Epochs"):
                    assert self.epochs is None
                    self.epochs = line_tokens[1]
                elif line.startswith("Loss"):
                    assert self.loss is None
                    self.loss = line_tokens[1]
                elif line.startswith("Spearmanr"):
                    assert self.spearmanr is None
                    self.spearmanr = line_tokens[1]
                elif line.startswith("Total time"):
                    assert self.total_time is None
                    self.total_time = line_tokens[2]
                elif line.startswith("Time per epoch"):
                    assert self.time_per_epoch is None
                    self.time_per_epoch = line_tokens[3]
                if line.startswith("RuntimeError:"):
                    self.oom = True
                    return
            if expected_length:
                assert(int(self.epochs) == expected_length)
            assert self.epochs is not None
            assert self.loss is not None
            assert self.spearmanr is not None
            assert self.total_time is not None
            assert self.time_per_epoch is not None

    def get_epochs(self):
        return self.epochs if not self.oom else '-'

    def get_loss(self):
        return self.loss if not self.oom else '-'

    def get_spearmanr(self):
        return self.spearmanr if not self.oom else '-'

    def get_total_time(self):
        return self.total_time if not self.oom else '-'

    def get_time_per_epoch(self):
        r'''
        Returns the time per epoch in ms
        '''
        return ("%.5f" % (1000.0 * float(self.time_per_epoch))) if not self.oom else '-'


num_epochs = 100
frameworks = ['pytorch', 'pytorch', 'tf', 'tf']
devices = ['cpu', 'cuda', 'cpu', 'cuda']
ns_lists = \
    [[str(i) for i in range(100, 4001, 100)]] * 4
methods = ['neuralsort', 'softsort']

res = dict()

for framework, device, ns in zip(frameworks, devices, ns_lists):
    for n in ns:
        for method in methods:
            filename = "./benchmark_results_%s/N_%s_%s/N_%s_%s_DEVICE_%s.txt" %\
                (framework, n, method, n, method, device)
            print("Processing " + str(filename))
            results_parser = BenchmarkSoftsortBackwardsResultsParser()
            results_parser.parse(filename, expected_length=int(num_epochs))
            epochs = results_parser.get_epochs()
            loss = results_parser.get_loss()
            spearmanr = results_parser.get_spearmanr()
            total_time = results_parser.get_total_time()
            time_per_epoch = results_parser.get_time_per_epoch()
            res[(framework, device, n, method, 'epochs')] = epochs
            res[(framework, device, n, method, 'loss')] = loss
            res[(framework, device, n, method, 'spearmanr')] = spearmanr
            res[(framework, device, n, method, 'total_time')] = total_time
            res[(framework, device, n, method, 'time_per_epoch')] = time_per_epoch


def get_times_for_device_framework_and_method(device, framework, method):
    times = []
    for n in ns:
        time = res[(framework, device, n, method, 'time_per_epoch')]
        if time == '-':
            break
        times.append(time)
    times = np.array(times)
    return times


ns = np.array([str(i) for i in range(100, 4001, 100)])

for device in ['cpu', 'cuda']:
    time_normalization = 1000 if device == 'cpu' else 1
    for framework in ['pytorch', 'tf']:
        times_neuralsort = get_times_for_device_framework_and_method(
            device=device,
            framework=framework,
            method='neuralsort')
        times_softsort = get_times_for_device_framework_and_method(
            device=device,
            framework=framework,
            method='softsort')
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        fontsize = 16
        ax1.plot(ns[:len(times_neuralsort)].astype('int'), times_neuralsort.astype('float') / time_normalization,
                 color='red', linestyle='--')
        ax1.plot(ns[:len(times_softsort)].astype('int'), times_softsort.astype('float') / time_normalization,
                 color='blue', linestyle='-')
        plt.xticks(rotation=70, fontsize=fontsize)
        ax1.set_xticks(ns.astype('int'))
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.xlabel(r'$n$', fontsize=fontsize)
        plt.xticks(range(200, 4001, 200), fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if device == 'cuda':
            plt.ylim(0, 150)
            plt.ylabel('time per epoch (ms)', fontsize=fontsize)
        else:
            plt.ylim(0, 30)
            plt.ylabel('time per epoch (s)', fontsize=fontsize)
        title = ""
        if framework == 'pytorch':
            title += 'Pytorch'
        elif framework == 'tf':
            title += 'TensorFlow'
        if device == 'cuda':
            title += ' GPU'
        elif device == 'cpu':
            title += ' CPU'
        # plt.title(title)  # Title should go in the figure latex caption
        plt.legend(['NeuralSort', 'SoftSort'], fontsize=fontsize)
        plt.tight_layout()
        plt.savefig('images/' + title.replace(' ', '_') + '_softsort')

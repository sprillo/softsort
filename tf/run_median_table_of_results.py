from statistics import mean


class RunMedianResultsParser:
    r'''
    Parses an individual results (i.e. log) file and stores the results.
    '''
    def __init__(self):
        self.average_loss = []
        self.val_set_correctly_identified = []
        self.val_set_mean_squared_error = []
        self.val_set_r2 = []
        self.val_set_spearmanr = []
        self.test_set_correctly_identified = -1
        self.test_set_mean_squared_error = -1
        self.test_set_r2 = -1
        self.test_set_spearmanr = -1

    def parse(self, file_path, expected_length=None):
        r'''
        :param file_path: path to the results (i.e. log) file
        '''
        with open(file_path) as file:
            for line in file:
                line_tokens = line.replace(',', '').replace('\n', '').split(' ')
                if line.startswith("Average loss"):
                    self.average_loss.append(line_tokens[2][:8])
                elif line.startswith("Validation set"):
                    self.val_set_correctly_identified.append(line_tokens[4][:8])
                    self.val_set_mean_squared_error.append(line_tokens[8][:8])
                    self.val_set_r2.append(line_tokens[10][:8])
                    self.val_set_spearmanr.append(line_tokens[12][:8])
                elif line.startswith("Test set"):
                    # print(line_tokens)
                    self.test_set_correctly_identified = line_tokens[4][:8]
                    self.test_set_mean_squared_error = line_tokens[8][:8]
                    self.test_set_r2 = line_tokens[10][:8]
                    self.test_set_spearmanr = line_tokens[12][:8]
            # print("file_path = %s" % file_path)
            if expected_length:
                assert(len(self.val_set_correctly_identified) == expected_length)
                assert(len(self.val_set_mean_squared_error) == expected_length)
                assert(len(self.val_set_r2) == expected_length)
                assert(len(self.val_set_spearmanr) == expected_length)
            # Validate data
            for list_name in ["average_loss", "val_set_correctly_identified", "val_set_mean_squared_error",
                              "val_set_r2", "val_set_spearmanr"]:
                for i, elem in enumerate(self.__dict__[list_name]):
                    try:
                        float(elem)
                    except ValueError:
                        print(f"path:\n{file_path}\n{i}: list {list_name} contains non-float: {elem}")
                        raise ValueError

    def get_val_set_correctly_identified(self):
        return self.val_set_correctly_identified[-1]

    def get_val_set_mean_squared_error(self):
        return self.val_set_mean_squared_error[-1]

    def get_val_set_r2(self):
        return self.val_set_r2[-1]

    def get_val_set_spearmanr(self):
        return self.val_set_spearmanr[-1]

    def get_test_set_correctly_identified(self):
        return self.test_set_correctly_identified

    def get_test_set_mean_squared_error(self):
        return self.test_set_mean_squared_error

    def get_test_set_r2(self):
        return self.test_set_r2

    def get_test_set_spearmanr(self):
        return self.test_set_spearmanr


num_epochs = '100'
l = '4'
m = '20'
ns = ['5', '9', '15']
methods = ['deterministic_neuralsort', 'stochastic_neuralsort', 'deterministic_softsort', 'stochastic_softsort']
lr = '0.001'
taus = [
    ['1024', '2048', '2048', '4096'],
    ['512', '512', '2048', '2048'],
    ['1024', '4096', '256', '2048']]
repetitions = ['0']

res = dict()

for n, taus_for_each_method in zip(ns, taus):
    for method, tau in zip(methods, taus_for_each_method):
        val_set_correctly_identified = []
        val_set_mean_squared_error = []
        val_set_r2 = []
        val_set_spearmanr = []
        test_set_correctly_identified = []
        test_set_mean_squared_error = []
        test_set_r2 = []
        test_set_spearmanr = []
        for repetition in repetitions:
            filename = "./run_median_results/N_%s_%s/N_%s_%s_TAU_%s_LR_%s_E_%s_REP_%s.txt" %\
                (n, method, n, method, tau, lr, num_epochs, repetition)
            # print("Processing " + str(filename))
            results_parser = RunMedianResultsParser()
            results_parser.parse(filename, expected_length=int(num_epochs))
            val_set_correctly_identified.append(float(results_parser.get_val_set_correctly_identified()))
            val_set_mean_squared_error.append(float(results_parser.get_val_set_mean_squared_error()))
            val_set_r2.append(float(results_parser.get_val_set_r2()))
            val_set_spearmanr.append(float(results_parser.get_val_set_spearmanr()))
            test_set_correctly_identified.append(float(results_parser.get_test_set_correctly_identified()))
            test_set_mean_squared_error.append(float(results_parser.get_test_set_mean_squared_error()))
            test_set_r2.append(float(results_parser.get_test_set_r2()))
            test_set_spearmanr.append(float(results_parser.get_test_set_spearmanr()))
        res[(n, tau, method, 'test_set_correctly_identified')] = mean(test_set_correctly_identified)
        res[(n, tau, method, 'test_set_mean_squared_error')] = mean(test_set_mean_squared_error)
        res[(n, tau, method, 'test_set_r2')] = mean(test_set_r2)
        res[(n, tau, method, 'test_set_spearmanr')] = mean(test_set_spearmanr)


def pretty_print_table(table):
    r'''
    Pretty prints the given table (of size (1 + #methods) x (1 + #ns))
    '''
    res = ""
    nrow = len(table)
    ncol = len(table[0])
    # Print header
    header = table[0]
    for c in range(ncol):
        if c == 0:
            res += "{:<31}".format('')
        else:
            res += "| n = " + "{:<10}".format(header[c])
    res += "\n"
    for r in range(1, nrow):
        # Method name
        res += "{:<31}".format(table[r][0])
        for c in range(1, ncol):
            res += "| " + table[r][c].replace('\\', '') + "  "
        res += "\n"
    print(res)


def pretty_print_table_latex(table):
    r'''
    Pretty prints the given table (of size (1 + #methods) x (1 + #ns))
    '''
    algorithm_names = {
        "deterministic_neuralsort": "Deterministic NeuralSort",
        "stochastic_neuralsort": "Stochastic NeuralSort",
        "deterministic_softsort": "Deterministic SoftSort",
        "stochastic_softsort": "Stochastic SoftSort"
    }
    res = "" \
          "\\begin{tabular}{lccc}\n" \
          "\\toprule\n"
    nrow = len(table)
    ncol = len(table[0])
    # Print header
    header = table[0]
    for c in range(ncol):
        if c == 0:
            res += "Algorithm "
        else:
            res += "& $n = " + "{}$ ".format(header[c])
    res += "\\\\\n"
    res += "\\midrule\n"
    for r in range(1, nrow):
        # Method name
        res += "{} ".format(algorithm_names[table[r][0]])
        for c in range(1, ncol):
            res += "& $" + table[r][c] + "$ "
        res += "\\\\\n"
    res += "" \
           "\\bottomrule\n" \
           "\\end{tabular}\n"
    print(res)


def print_table(latex=False):
    table = []
    # Add table header
    header = ["algorithm"] + [n for n in ns]
    table.append(header)
    for i, method in enumerate(methods):
        row = [method]
        for j, n in enumerate(ns):
            tau = taus[j][i]
            test_set_mean_squared_error = res[(n, tau, method, 'test_set_mean_squared_error')]
            test_set_spearmanr = res[(n, tau, method, 'test_set_spearmanr')]
            table_entry = "%.2f\\ (%.2f)" % (test_set_mean_squared_error * 1e-4, test_set_spearmanr)
            row.append(table_entry)
        table.append(row)
    if latex:
        pretty_print_table_latex(table)
    else:
        pretty_print_table(table)


print_table(latex=True)
print_table(latex=False)

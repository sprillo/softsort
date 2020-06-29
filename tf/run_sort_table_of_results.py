from statistics import mean, median, stdev


class RunSortResultsParser:
    r'''
    Parses an individual results (i.e. log) file and stores the results.
    '''
    def __init__(self):
        self.average_loss = []
        self.val_set_prop_all_correct = []
        self.val_set_prop_any_correct = []
        self.test_set_prop_all_correct = -1
        self.test_set_prop_any_correct = -1

    def parse(self, file_path, expected_length=None):
        r'''
        :param file_path: path to the results (i.e. log) file
        '''
        with open(file_path) as file:
            for line in file:
                line_tokens = line.replace(',', ' ').replace('\n', '').split(' ')
                if line.startswith("Average loss"):
                    self.average_loss.append(line_tokens[2][:8])
                elif line.startswith("Validation set"):
                    self.val_set_prop_all_correct.append(line_tokens[5][:8])
                    self.val_set_prop_any_correct.append(line_tokens[10][:8])
                elif line.startswith("Test set"):
                    self.test_set_prop_all_correct = line_tokens[5][:8]
                    self.test_set_prop_any_correct = line_tokens[10][:8]
            # print("file_path = %s" % file_path)
            if expected_length:
                assert(len(self.val_set_prop_all_correct) == expected_length)
                assert(len(self.val_set_prop_any_correct) == expected_length)
            # Check that all parsed entries are floats
            for list_name in ["average_loss", "val_set_prop_all_correct", "val_set_prop_any_correct"]:
                for i, elem in enumerate(self.__dict__[list_name]):
                    try:
                        float(elem)
                    except ValueError:
                        print(f"path:\n{file_path}\n{i}: list {list_name} contains non-float: {elem}")
                        raise ValueError

    def get_val_set_prop_all_correct(self):
        return self.val_set_prop_all_correct[-1]

    def get_val_set_prop_any_correct(self):
        return self.val_set_prop_any_correct[-1]

    def get_test_set_prop_all_correct(self):
        return self.test_set_prop_all_correct

    def get_test_set_prop_any_correct(self):
        return self.test_set_prop_any_correct


num_epochs = '100'
l = '4'
m = '20'
ns = ['3', '5', '7', '9', '15']
methods = ['deterministic_neuralsort', 'stochastic_neuralsort', 'deterministic_softsort', 'stochastic_softsort']
lr = '0.005'
taus = ['1024', '1024', '1024', '128', '128']
repetitions = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

res = dict()

for n, tau in zip(ns, taus):
    for method in methods:
        val_set_all_correct = []
        val_set_any_correct = []
        test_set_all_correct = []
        test_set_any_correct = []
        for repetition in repetitions:
            filename = "./run_sort_results/N_%s_%s/N_%s_%s_TAU_%s_LR_%s_E_%s_REP_%s.txt" %\
                (n, method, n, method, tau, lr, num_epochs, repetition)
            # print("Processing " + str(filename))
            results_parser = RunSortResultsParser()
            results_parser.parse(filename, expected_length=int(num_epochs))
            val_set_all_correct.append(float(results_parser.get_val_set_prop_all_correct()))
            val_set_any_correct.append(float(results_parser.get_val_set_prop_any_correct()))
            test_set_all_correct.append(float(results_parser.get_test_set_prop_all_correct()))
            test_set_any_correct.append(float(results_parser.get_test_set_prop_any_correct()))
        res[(n, tau, method, 'test_set_all_correct_median')] = median(test_set_all_correct)
        res[(n, tau, method, 'test_set_any_correct_median')] = median(test_set_any_correct)
        res[(n, tau, method, 'test_set_all_correct_mean')] = mean(test_set_all_correct)
        res[(n, tau, method, 'test_set_any_correct_mean')] = mean(test_set_any_correct)
        # Add SDs
        res[(n, tau, method, 'test_set_all_correct_sd')] = stdev(test_set_all_correct)
        res[(n, tau, method, 'test_set_any_correct_sd')] = stdev(test_set_any_correct)


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
            res += "| n = " + "{:<11}".format(header[c])
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
          "\\begin{tabular}{lccccc}\n" \
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


def print_table_for_metric(
        mean_or_median,
        legacy=False,  # Prints as in SoftSort paper v0 (which is same as NeuralSort and OT paper)
        show_test_set_all_correct=True,  # Ignored if legacy=True
        test_set_all_any_correct=False,  # Ignored if legacy=True
        latex=False):
    print('*' * 30 + (' %s over %d runs ' % (mean_or_median, len(repetitions))) + '*' * 30)
    table = []
    # Add table header
    header = ["algorithm"] + [n for n in ns]
    table.append(header)
    for method in methods:
        row = [method]
        for n, tau in zip(ns, taus):
            test_set_all_correct = res[(n, tau, method, 'test_set_all_correct_' + mean_or_median)]
            test_set_all_correct_sd = res[(n, tau, method, 'test_set_all_correct_sd')]
            test_set_any_correct = res[(n, tau, method, 'test_set_any_correct_' + mean_or_median)]
            test_set_any_correct_sd = res[(n, tau, method, 'test_set_any_correct_sd')]
            if legacy:
                table_entry = "%.3f\\ (%.3f)" % (test_set_all_correct, test_set_any_correct)
            else:
                table_entry = ""
                if show_test_set_all_correct:
                    table_entry += "%.3f\\ \\pm\\ %.3f" % (test_set_all_correct, test_set_all_correct_sd)
                if test_set_all_any_correct:
                    table_entry += "%.3f\\ \\pm\\ %.3f" % (test_set_any_correct, test_set_any_correct_sd)
            row.append(table_entry)
        table.append(row)
    if latex:
        pretty_print_table_latex(table)
    else:
        pretty_print_table(table)


print('*' * 90)
print('*' * 30 + ' Printing Legacy Tables (no SDs) ' + '*' * 30)
print('*' * 90)
print_table_for_metric('mean', legacy=True, latex=True)
print_table_for_metric('mean', legacy=True, latex=False)

print('*' * 90)
print('*' * 30 + ' Printing New Tables with SDs ' + '*' * 30)
print('*' * 90)

for show_metric in [(True, False), (False, True)]:
    print_table_for_metric(
        'mean',
        legacy=False,
        show_test_set_all_correct=show_metric[0],
        test_set_all_any_correct=show_metric[1],
        latex=True)

    print_table_for_metric(
        'mean',
        legacy=False,
        show_test_set_all_correct=show_metric[0],
        test_set_all_any_correct=show_metric[1],
        latex=False)

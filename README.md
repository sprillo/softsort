This repository is a fork of ```ermongroup/neuralsort``` implementing the SoftSort operator and reproducing all the results reported in the paper "SoftSort: A Continuous Relaxation for the argsort Operator".

## Requirements

The codebase is implemented in Python 3.7. To install the necessary requirements, run the following commands:

```
pip3 install -r requirements.txt
```

## Sorting Handwritten Numbers Experiment

To reproduce the results in Table 1, just run:

```
cd tf
bash run_sort.sh
python3 run_sort_table_of_results.py
```

The first script (bash) will train all models. This takes a long time. You can inspect this script to see what parameters were used to train each model (which are the ones reported in the paper). The second script (python) will process the results from the models and print Table 1.

To train a single model directly, you can use the `tf/run_sort.py` script, with the following arguments:

```
  --M INT                 Minibatch size
  --n INT                 Number of elements to compare at a time
  --l INT                 Number of digits in each multi-mnist dataset element
  --tau FLOAT             Temperature (either of sinkhorn or neuralsort relaxation)
  --method STRING         One of 'deterministic_neuralsort', 'stochastic_neuralsort', 'deterministic_softsort', 'stochastic_softsort'
  --n_s INT               Number of samples for stochastic methods
  --num_epochs INT        Number of epochs to train
  --lr FLOAT              Initial learning rate
```

## Quantile Regression Experiment

To reproduce the results in Table 2, just run:

```
cd tf
bash run_median.sh
python3 run_median_table_of_results.py
```

The first script (bash) will train all models. This takes a long time. You can inspect this script to see what parameters were used to train each model (which are the ones reported in the paper). The second script (python) will process the results from the models and print Table 2.

To train a single model directly, you can use the `tf/run_median.py` script, with the following arguments:

```
  --M INT                 Minibatch size
  --n INT                 Number of elements to compare at a time
  --l INT                 Number of digits in each multi-mnist dataset element
  --tau FLOAT             Temperature (either of sinkhorn or neuralsort relaxation)
  --method STRING         One of 'deterministic_neuralsort', 'stochastic_neuralsort', 'deterministic_softsort', 'stochastic_softsort'
  --n_s INT               Number of samples for stochastic methods
  --num_epochs INT        Number of epochs to train
  --lr FLOAT              Initial learning rate
```

## Differentiable kNN Experiment

To reproduce the results in Table 3, run:

```
cd pytorch
bash run_dknn.sh
python3 run_dknn_table_of_results.py
```

The first script (bash) will train all the models. This takes about two days to sequentally test the different hyperparameter configurations. The seconds script iterates through logs and prints the best results.

To train a single model directly, you can use the `pytorch/run_dknn.py` script, with the following arguments:

```
  --simple                Whether to use our softsort, or the baseline neuralsort
  --k INT                 Number of nearest neighbors
  --tau FLOAT             Temperature of sorting operator
  --nloglr FLOAT          Negative log10 of learning rate
  --method STRING         One of 'deterministic', 'stochastic'
  --dataset STRING        One of 'mnist', 'fashion-mnist', 'cifar10'
  --num_train_queries INT Number of queries to evaluate during training.
  --num_train_neighbors INT Number of neighbors to consider during training.
  --num_samples INT       Number of samples for stochastic methods
  --num_epochs INT        Number of epochs to train
```

## Speed Comparison Experiment

To reproduce the results in Figure 6, just run:

```
bash synthetic_experiment_speed_comparison.sh
python3 synthetic_experiment_speed_comparison_plot.py
```

The first script (bash) will train all models. This takes some time. The second script (python) will process the results and print the graphs in Figure 6 under the ```images/``` directory.

## Learning Curves

For the synthetic experiment learning curves, run:

```
bash synthetic_experiment_learning_curves.sh
```

Then, to generate the plot in Figure 8, run:

```
python3 synthetic_experiment_learning_curves_plot.py
```

To generate the run_sort and run_median learning curve plots (Figure 7), run:

```
cd tf
python3 run_sort_learning_curves.py
python3 run_median_learning_curves.py
```

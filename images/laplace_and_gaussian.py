import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm


def plot(
        x_min,
        x_max,
        s_xs,
        s_ticks,
        distribution,
        legend,
        ylabel):
    fontsize = 16
    x = np.arange(start=x_min, stop=x_max, step=0.01)
    y = distribution.pdf(x)
    plt.plot(x, y)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.yticks([], [])
    plt.xticks(s_xs, s_ticks, fontsize=fontsize)
    plt.fill_between(x, y, facecolor='blue', alpha=0.2)
    for s_x in s_xs:
        plt.vlines(x=s_x, ymin=0, ymax=distribution.pdf(s_x))
    plt.legend([legend], fontsize=fontsize)
    plt.tight_layout()


plt.figure(figsize=(13, 5))

plt.subplot(1, 2, 1)
plot(x_min=-4,
     x_max=4,
     s_xs=[-2, 0, 1, 2.5],
     s_ticks=[r'$s_4$', '$s_3$', '$s_2$', '$s_1$'],
     distribution=laplace,
     legend=r'$\propto\phi_{Laplace(s_3, \tau)}$',
     ylabel=r'$SoftSort^{|\cdot|}_\tau(s)[3, :]$')

plt.subplot(1, 2, 2)
plot(x_min=-4,
     x_max=4,
     s_xs=[-1.2, 0, 1.2, 2.4],
     s_ticks=[r'$s_4$', '$s_3$', '$s_2$', '$s_1$'],
     distribution=norm,
     legend=r'$\propto\phi_{\mathcal{N}(s_3, a\tau)}$',
     ylabel=r'$NeuralSort_\tau(s)[3, :]$')

plt.savefig('laplace_and_gaussian_softsort')

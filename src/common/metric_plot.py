from matplotlib import pyplot as plt
import numpy as np
def metric_plot(plt_ndkl_before,plt_ndkl_after,baseline,metric_name):
    # change line 5 and 6 with different values, for value x since they need to be coded dynamically later with a lambda function
    plt.scatter(np.arange(len(plt_ndkl_before)),plt_ndkl_before,label=f'{metric_name} before')
    plt.scatter(np.arange(len(plt_ndkl_before)),plt_ndkl_after,label=f'{metric_name} after')
    plt.legend()
    plt.savefig(f'{baseline}metric-{metric_name}.png')
    plt.show()

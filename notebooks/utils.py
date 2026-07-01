########################################################################

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple
from torch import Tensor

## function given with notebook
def plot_pyloric(ts: Union[ndarray, Tensor], v: Union[ndarray, Tensor], axs: Optional[plt.Axes] = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the voltage of the pyloric network for each neuron.

    Args:
        ts: The time points to plot.
        v: The voltage of the pyloric network.
        axs: The axes to plot on. Allows to plot multiple traces in one figure.

    Returns:
        fig: The figure.
        axs: The axes.
    """
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(10, 5), sharex=True, layout='constrained')
    for ax_i, v_i in zip(axs, v):
        ax_i.plot(ts, v_i, **kwargs)
        ax_i.set_ylabel('V (mV)')
    axs[0].set_title(f'AB/PD')
    axs[1].set_title(f'LP')
    axs[2].set_title(f'PY')
    axs[2].set_xlabel('t (ms)')
    return fig, axs
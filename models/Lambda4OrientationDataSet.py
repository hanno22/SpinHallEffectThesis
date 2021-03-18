import numpy as np


class Lambda4OrientationDataSet:
    def __init__(self, data_path):
        self.data = np.genfromtxt(data_path, delimiter=',')

    def plot_orientation(self, fig, ax):
        ax.set_ylabel('Power in mW')
        ax.plot(self.data[:, 0], self.data[:, 1], "b--", label="power after polarisation filter")
        ax.tick_params(axis='y')

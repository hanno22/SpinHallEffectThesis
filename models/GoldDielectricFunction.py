import numpy as np
import scipy.constants as con


class GoldDielectricFunction:
    def __init__(self, data_path):
        data = np.genfromtxt(data_path,
                             skip_header=1,
                             # skip_footer=1,
                             # names=True,
                             # dtype=None,
                             delimiter='	')
        self.E = data[:, 0]
        self.omega = self.E * con.e / con.hbar
        #self.wavelength = data[:, 1]
        self.eps = data[:, 2] + 1j * data[:, 3]
        self.n = data[:, 4]
        self.k = data[:, 5]
        self.k_0 = self.omega / con.speed_of_light

    def k_spp(self, eps_d):
        return self.omega / con.c * np.sqrt(eps_d * self.eps / (eps_d + self.eps))

    def plot_k_spp(self, eps_d, fig, ax):
        k_spp = self.k_spp(eps_d) * 1e-6
        color = next(ax._get_lines.prop_cycler)['color']
        ax.scatter(k_spp.real, self.E, color=color, marker='.', label='$\\epsilon_D = {:.2f}$'.format(eps_d))
        ax.plot(self.k_0 * np.sqrt(eps_d) * 1e-6 , self.E, color=color, linestyle='--')


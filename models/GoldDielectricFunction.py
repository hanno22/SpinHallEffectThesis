import numpy as np
import scipy.constants as con
from scipy.interpolate import interp1d

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
        # self.wavelength = data[:, 1]
        self.eps = data[:, 2] + 1j * data[:, 3]
        self.n = data[:, 4]
        self.k = data[:, 5]
        self.k_0 = self.omega / con.speed_of_light
        self.real_interpolated = interp1d(self.E, self.eps.real)
        self.imag_interpolated = interp1d(self.E, self.eps.imag)

    def eps_interpolated(self, energy):
        return self.real_interpolated(energy) + 1j * self.imag_interpolated(energy)

    def k_spp(self, eps_d):
        return self.omega / con.c * self.n_eff(eps_d)

    def n_eff(self, eps_d):
        return np.lib.scimath.sqrt(eps_d * self.eps / (eps_d + self.eps))

    def plot_k_spp(self, eps_d, color, fig, ax):
        k_spp = self.k_spp(eps_d) * 1e-6
        #color = next(ax._get_lines.prop_cycler)['color']
        ax.scatter(k_spp.real, self.E, color=color, marker='.', label='$k_\\mathrm{spp}(E)$')
        self.plot_light_line(eps_d, color, '$k_{0}$', fig, ax)
    def plot_light_line(self, eps_d, color, label, fig, ax):
        ax.plot(self.k_0 * np.sqrt(eps_d) * 1e-6, self.E, color=color, linestyle='--', label=label)

    def full_plot(self, fig, ax):
        self.plot_k_spp(1.0 ** 2,'tab:blue', fig, ax)
        #color = next(ax._get_lines.prop_cycler)['color']
        self.plot_light_line(1.52 ** 2, 'tab:orange', '$k_0 \\sqrt{\\epsilon_\\mathrm{Glas}}$', fig, ax)
        ax.axhline(con.c * con.h / 633e-9 / con.eV, ls='-.', color='k', label="E(633nm)")
        ax.legend(loc='upper right')
        # ax.grid()
        # inset axes....
        axins = ax.inset_axes([0.6, 0.03, 0.5, 0.5])
        #self.plot_k_spp(1.52 ** 2, fig, axins)
        self.plot_k_spp(1 ** 2, 'tab:blue', fig, axins)
        self.plot_light_line(1.52 ** 2, 'tab:orange', '', fig, axins)
        axins.axhline(con.c * con.h / 633e-9 / con.e, ls='-.', color='k', label="E(633nm)")
        # sub region of the original image
        x1, x2, y1, y2 = 8.7, 16, 1.3, 2.6
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        ax.indicate_inset_zoom(axins)
        ax.set(xlabel='$\\operatorname{\\mathbb{R}e}\\left\\{k\\right\\} / \mu m^{-1}$', ylabel='E / eV')
        fig.tight_layout()

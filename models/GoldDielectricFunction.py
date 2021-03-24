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

    def full_plot(self, fig, ax):
        self.plot_k_spp(1.52 ** 2, fig, ax)
        self.plot_k_spp(1 ** 2, fig, ax)
        ax.axhline(con.c * con.h / 633e-9 / con.eV, ls='-.', color='k', label="E(633nm)")
        ax.legend()
        # ax.grid()
        # inset axes....
        axins = ax.inset_axes([0.6, 0.03, 0.5, 0.5])
        self.plot_k_spp(1.52 ** 2, fig, axins)
        self.plot_k_spp(1 ** 2, fig, axins)
        axins.axhline(con.c * con.h / 633e-9 / con.e, ls='-.', color='k', label="E(633nm)")
        # sub region of the original image
        x1, x2, y1, y2 = 9, 18, 1.3, 2.6
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        ax.indicate_inset_zoom(axins)
        ax.set(xlabel='$\Re(k_{spp}) / \mu m^{-1}$', ylabel='E / eV')


import cmath as cm
import numpy as np


class JonesVector(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj.view(cls)

    @property
    def delta(self):
        return cm.phase(self[1]) - cm.phase(self[0])

    @property
    def alpha(self):
        return 1 / 2 * np.arctan(2 * abs(self[0]) * abs(self[1]) * np.cos(self.delta) / (
                abs(self[0]) ** 2 - abs(self[1]) ** 2))

    @property
    def a(self):
        return np.sqrt((abs(self[0]) * np.cos(self.alpha)) ** 2 + (
                abs(self[1]) * np.sin(self.alpha)) ** 2
                       + abs(self[0]) * abs(self[1]) * np.cos(self.delta) * np.sin(
            2 * self.alpha))

    @property
    def b(self):
        return np.sqrt((abs(self[0]) * np.sin(self.alpha)) ** 2 + (
                abs(self[1]) * np.cos(self.alpha)) ** 2
                       - abs(self[0]) * abs(self[1]) * np.cos(self.delta) * np.sin(
            2 * self.alpha))

    @property
    def ellipticity(self):
        if self.a > self.b:
            return self.b / self.a
        else:
            return self.a / self.b

    @property
    def intensity(self):
        return np.absolute(self) ** 2

    def plot_ellipsis(self, fig, ax):
        phases = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
        x = (phases * self[0]).real
        y = (phases * self[1]).real
        a = self.a
        b = self.b
        alpha = self.alpha
        x_alpha = a * np.array([np.cos(alpha + np.pi), np.cos(alpha)])
        y_alpha = a * np.array([np.sin(alpha + np.pi), np.sin(alpha)])
        x_alpha_pi_2 = b * np.array([np.cos(alpha + np.pi / 2 + np.pi), np.cos(alpha + np.pi / 2)])
        y_alpha_pi_2 = b * np.array([np.sin(alpha + np.pi / 2 + np.pi), np.sin(alpha + np.pi / 2)])

        ax.axis('equal')
        ax.set(xlabel='$E_x$ / AU', ylabel='$E_y$ / AU')
        ax.grid()
        ax.plot(x, y, label='$\\vec{E}(t) = \\operatorname{\\mathbb{R}e}\\left\\{\\vec{J}\\exp\\left(i \\omega t\\right)\\right\\}$')
        ax.plot(x_alpha, y_alpha, 'r--', label='$a,b$')
        ax.plot(x_alpha_pi_2, y_alpha_pi_2, 'r--')
        ax.legend()


class JonesMatrix(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj.view(cls)

    def rotate(self, angle):
        def rotation_matrix(a):
            return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

        return rotation_matrix(-angle) @ self @ rotation_matrix(-angle)

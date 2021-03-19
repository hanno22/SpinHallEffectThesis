import numpy as np
from scipy.optimize import least_squares
import cmath as cm
from matplotlib.patches import Ellipse


class Lambda4OrientationDataSet:
    def __init__(self, data_path):
        data = np.genfromtxt(data_path, delimiter=',')
        self.retarder_angles = np.deg2rad(data[:, 0])
        self.measured_intensities = data[:, 1]

        self.retarder = np.exp(-1j * np.pi / 4) * np.mat([[1, 0],
                                                          [0, 1j]])
        self.diattenuator = np.array([[1, 0],
                                      [0, 0]])

    def plot_orientation(self, fig, ax):
        ax.set_ylabel('Power in mW')
        ax.plot(self.angles_r, self.measured_intensities, "b--", label="power after polarisation filter")
        ax.tick_params(axis='y')

    def __intensity_diff(self, parameters):
        Ex, Ex_j, Ey, Ey_j, dia_offset = parameters;
        jones_vector_model = np.array([Ex + 1j * Ex_j, Ey + 1j * Ey_j])
        model_jones_vectors = self.polarimeter_simulation(jones_vector_model, dia_offset)
        return self.measured_intensities - self.__get_intensities(model_jones_vectors)

    def fit(self, tol=1e-12):
        sup_limit = np.array(np.array([10, 10, 10, 10, np.pi]))
        inf_limit = np.array(np.array([-10, -10, -10, -10, -np.pi]))
        x_0 = (1, 0, 0, 0, 0)  # np.random.rand(4) * sup_limit
        result = least_squares(fun=self.__intensity_diff, x0=x_0, bounds=(inf_limit, sup_limit), ftol=tol, xtol=tol,
                               gtol=tol)
        # result = leastsq(self.__intensity_diff, x_0)
        Ex, Ex_j, Ey, Ey_j, dia_offset = result.x
        jones_vector = np.array([Ex + 1j * Ex_j, Ey + 1j * Ey_j])
        return jones_vector, dia_offset

    def polarimeter_simulation(self, jones_vector, dia_offset):
        #dia_offset = 0
        jones_vector_results = np.zeros((*self.retarder_angles.shape, *jones_vector.shape), dtype='D')
        rotated_diattenuator = self.__rotate_matrix(self.diattenuator, dia_offset)
        for idx, angle in enumerate(self.retarder_angles):
            rotated_retarders = self.__rotate_matrix(self.retarder, angle)
            jones_vector_results[idx] = rotated_diattenuator @ rotated_retarders @ jones_vector
        return jones_vector_results

    def plot_jones_vector_ellipse(self, jones_vector, fig, ax):
        phases = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
        x = (phases * jones_vector[0]).real
        y = (phases * jones_vector[1]).real

        delta = cm.phase(jones_vector[1]) - cm.phase(jones_vector[0])
        alpha = 1 / 2 * np.arctan(2 * abs(jones_vector[0]) * abs(jones_vector[1]) * np.cos(delta) / (
                abs(jones_vector[0]) ** 2 - abs(jones_vector[1]) ** 2))
        E_alpha = np.sqrt((abs(jones_vector[0]) * np.cos(alpha)) ** 2 + (abs(jones_vector[1]) * np.sin(alpha)) ** 2
                          + abs(jones_vector[0]) * abs(jones_vector[1]) * np.cos(delta) * np.sin(2 * alpha))
        E_alpha_pi_2 = np.sqrt((abs(jones_vector[0]) * np.sin(alpha)) ** 2 + (abs(jones_vector[1]) * np.cos(alpha)) ** 2
                          - abs(jones_vector[0]) * abs(jones_vector[1]) * np.cos(delta) * np.sin(2 * alpha))
        ellipticity = abs(E_alpha_pi_2/E_alpha)

        x_alpha = E_alpha * np.array([np.cos(alpha + np.pi), np.cos(alpha)])
        y_alpha = E_alpha * np.array([np.sin(alpha + np.pi), np.sin(alpha)])
        x_alpha_pi_2 = E_alpha_pi_2 * np.array([np.cos(alpha + np.pi/2 + np.pi), np.cos(alpha+ np.pi /2)])
        y_alpha_pi_2 = E_alpha_pi_2 * np.array([np.sin(alpha + np.pi/2 + np.pi), np.sin(alpha+ np.pi /2)])
        # ax.axis('square')
        ax.plot(x_alpha, y_alpha, 'r--')
        ax.plot(x_alpha_pi_2, y_alpha_pi_2, 'r--')
        ax.axis('equal')
        ax.set_xlabel('$E_x$')
        ax.set_ylabel('$E_y$')
        ax.grid()
        ax.text(0.5, 0.9, '$\\delta$={:.1f}°, $\\alpha$={:.1f}°, ellipticity={:.1f}'.format(np.rad2deg(delta), np.rad2deg(alpha), ellipticity),
                ha='center', va='center', transform=ax.transAxes)
        ax.plot(x, y)

    def plot_jones_fit(self, jones_vector, dia_offset, fig, ax):
        ax.plot(self.retarder_angles, self.measured_intensities)

        sim_vectors = self.polarimeter_simulation(jones_vector, dia_offset)
        sim_int = self.__get_intensities(sim_vectors)

        ax.plot(self.retarder_angles, sim_int, '--')

    def __get_intensities(self, jones_vectors):
        return np.sum(np.absolute(jones_vectors) ** 2, axis=1)

    def __rotation_matrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    def __rotate_matrix(self, matrix, angle):
        return self.__rotation_matrix(-angle) @ matrix @ self.__rotation_matrix(-angle)

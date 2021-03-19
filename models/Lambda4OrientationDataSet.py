import numpy as np
from scipy.optimize import least_squares


class Lambda4OrientationDataSet:
    def __init__(self, data_path):
        data = np.genfromtxt(data_path, delimiter=',')
        self.retarder_angles = np.deg2rad(data[:, 0])
        self.measured_intensities = data[:, 1]

        self.retarder = np.exp(-1j * np.pi / 4) * np.mat([[1, 0],
                                                          [0, 1j]])
        self.diattenuator = np.array([[1, 0],
                                      [0, 0]])
        self.rotated_retarders = np.zeros((*self.retarder_angles.shape, *self.retarder.shape), dtype='D')
        self.rotated_diattenuators = np.zeros((*self.retarder_angles.shape, *self.diattenuator.shape), dtype='D')
        for idx, angle in enumerate(self.retarder_angles):
            self.rotated_retarders[idx] = np.matmul(self.__rotation_matrix(-angle),
                                                    np.matmul(self.retarder, self.__rotation_matrix(-angle)))
            self.rotated_diattenuators[idx] = self.diattenuator  # do not rotate diattenuator


    def plot_orientation(self, fig, ax):
        ax.set_ylabel('Power in mW')
        ax.plot(self.angles_r, self.measured_intensities, "b--", label="power after polarisation filter")
        ax.tick_params(axis='y')

    def __intensity_diff(self, parameters):
        Ex, Ex_j, Ey, Ey_j = parameters;
        jones_vector_model = np.array([Ex + 1j * Ex_j, Ey + 1j * Ey_j])
        model_jones_vectors = self.polarimeter_simulation(jones_vector_model)
        return self.measured_intensities - self.__get_intensities(model_jones_vectors)

    def fit(self, tol=1e-12):
        sup_limit = np.array(np.array([2, 2, 2, 2]))
        inf_limit = np.array(np.array([-2, -2, -2, -2]))
        x_0 = np.random.rand(4) * sup_limit
        result = least_squares(fun=self.__intensity_diff, x0=x_0, bounds=(inf_limit, sup_limit), ftol=tol, xtol=tol, gtol=tol)
        #result = leastsq(self.__intensity_diff, x_0)
        return result

    def polarimeter_simulation(self, jones_vector):
        #jones_vector = np.array([Ex + 1j * Ex_j, Ey + 1j * Ey_j])
        jones_vector_results = np.zeros((*self.retarder_angles.shape, *jones_vector.shape), dtype='D')
        for idx, angle in enumerate(self.retarder_angles):
            jones_vector_results[idx] = np.matmul(self.rotated_diattenuators[idx],
                                                  np.matmul(self.rotated_retarders[idx],
                                                            jones_vector))
        return jones_vector_results

    def plot_jones_vector_ellipse(self, jones_vector, fig, ax):
        phases = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
        x = (phases * jones_vector[0]).real
        y = (phases * jones_vector[1]).real
        ax.plot(x, y)
    def plot_jones_fit(self, jones_vector, fig, ax):
        ax.plot(self.retarder_angles, self.measured_intensities)
        sim_vectors = self.polarimeter_simulation(jones_vector)
        sim_int = self.__get_intensities(sim_vectors)
        ax.plot(self.retarder_angles, sim_int, '--')

    def __get_intensities(self, jones_vectors):
        return np.sum(np.absolute(jones_vectors) ** 2, axis=1)

    def __rotation_matrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])


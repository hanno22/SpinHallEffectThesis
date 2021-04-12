import numpy as np
from scipy.optimize import least_squares

from models.JonesFormalism import JonesVector, JonesMatrix


class Lambda4OrientationDataSet:
    def __init__(self, data_path):
        data = np.genfromtxt(data_path, delimiter=',')
        self.retarder_angles = np.deg2rad(data[:, 0])
        self.measured_intensities = data[:, 1]
        self.retarder = np.exp(-1j * np.pi / 4) * JonesMatrix([[1, 0], [0, 1j]])
        self.diattenuator = JonesMatrix([[1, 0], [0, 0]])

    def plot_orientation(self, fig, ax):
        ax.set_ylabel('Power in mW')
        ax.plot(np.rad2deg(self.retarder_angles), self.measured_intensities, "b--",
                label="power after polarisation filter")
        ax.tick_params(axis='y')

    def __intensity_diff(self, parameters):
        Ex, Ex_j, Ey, Ey_j, dia_offset = parameters
        jones_vector_model = JonesVector([Ex + 1j * Ex_j, Ey + 1j * Ey_j])
        model_jones_vectors = self.polarimeter_simulation(jones_vector_model, dia_offset)
        return self.measured_intensities - self.__get_intensities(model_jones_vectors)

    def fit(self, tol=1e-12):
        sup_limit = np.array([10, 10, 10, 10, np.pi])
        inf_limit = np.array([-10, -10, -10, -10, -np.pi])
        x_0 = (1, 0, 0, 0, 0)  # np.random.rand(4) * sup_limit
        result = least_squares(fun=self.__intensity_diff, x0=x_0, bounds=(inf_limit, sup_limit), ftol=tol, xtol=tol,
                               gtol=tol)
        # result = leastsq(self.__intensity_diff, x_0)
        Ex, Ex_j, Ey, Ey_j, dia_offset = result.x
        jones_vector = JonesVector([Ex + 1j * Ex_j, Ey + 1j * Ey_j])
        return jones_vector, dia_offset

    def polarimeter_simulation(self, jones_vector, dia_offset):
        #dia_offset = 0
        jones_vector_results = np.zeros((*self.retarder_angles.shape, *jones_vector.shape), dtype='D')
        rotated_diattenuator = self.diattenuator.rotate(dia_offset)
        for idx, angle in enumerate(self.retarder_angles):
            rotated_retarders = self.retarder.rotate(angle)
            jones_vector_results[idx] = rotated_diattenuator @ rotated_retarders @ jones_vector
            # Todo matmul of JonesMatrix with jones_vector should return JonesVector
        return jones_vector_results


    def plot_jones_fit(self, jones_vector, dia_offset, fig, ax):
        ax.plot(self.retarder_angles, self.measured_intensities, '+', label="measured")
        sim_vectors = self.polarimeter_simulation(jones_vector, dia_offset)
        sim_int = self.__get_intensities(sim_vectors)
        ax.set_ylabel('Power in mW')
        ax.set_xlabel('$\\lambda/4$-wave plate orientation in rad')
        ax.plot(self.retarder_angles, sim_int, '--', label="model")

    def __get_intensities(self, jones_vectors):
        return np.sum(np.absolute(jones_vectors) ** 2, axis=1)

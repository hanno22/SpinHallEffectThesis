import numpy as np
import cv2 as cv
import glob
import os
import re
from scipy.optimize import least_squares


class SpinHallDataSetFp:

    def __init__(self, data_path, center, max_radius, fp_scale, lamb_offset=0):
        self.center = center
        self.max_radius = max_radius
        self.fp_scale = fp_scale

        def get_angle_from_file_name(file_name):
            return float(re.search("([0-9]*).bmp", file_name).group(1))

        files = sorted(glob.glob(os.path.join(data_path, '[0-9]*.bmp')),
                       key=get_angle_from_file_name)
        self.lambda_4_angles = np.empty(len(files))
        self.__data_dimensions = (max_radius * 2, max_radius * 2)
        self.data = np.empty((len(files), *self.__data_dimensions))
        self.polar_data = np.empty_like(self.data)

        for i, file in enumerate(files):
            self.data[i] = cv.cvtColor(cv.imread(file), cv.COLOR_BGR2GRAY)[self.center[1] - self.max_radius:self.center[1] + self.max_radius, self.center[0] - self.max_radius:self.center[0] + self.max_radius]
            self.lambda_4_angles[i] = get_angle_from_file_name(file)
            self.polar_data[i] = cv.linearPolar(self.data[i], (max_radius, max_radius), max_radius, cv.WARP_FILL_OUTLIERS, cv.INTER_CUBIC)

        self.lambda_4_angles = np.where(self.lambda_4_angles - lamb_offset > 0, self.lambda_4_angles - lamb_offset,
                                        360 + self.lambda_4_angles - lamb_offset)

        self.radial_values = np.linspace(0, self.max_radius, self.__data_dimensions[1])
        self.radial_values_k = np.linspace(0, self.max_radius / self.fp_scale, self.__data_dimensions[1])
        self.angular_values = np.linspace(0, 2 * np.pi, self.__data_dimensions[0])

    def __get_lambda_4_index(self, lambda_4_angle):
        idx = (np.abs(self.lambda_4_angles - lambda_4_angle)).argmin()
        return self.lambda_4_angles[idx], idx

    def __get_angle_masks(self, mid_angle, angle_width, angle_gap):
        if mid_angle + angle_width < 2 * np.pi:
            if mid_angle + angle_gap < 2 * np.pi:
                left_angle_mask = (self.angular_values >= mid_angle + angle_gap) & (
                        self.angular_values <= mid_angle + angle_width)
            else:
                raise Exception('something Wrong: mid_angle + remove_angs > 2 * np.pi:')
        else:
            if mid_angle + angle_gap < 2 * np.pi:
                left_angle_mask = (self.angular_values >= mid_angle + angle_gap) | (
                        self.angular_values <= mid_angle + angle_width - (2 * np.pi))
            else:
                left_angle_mask = (self.angular_values >= mid_angle + angle_gap - 2 * np.pi) & (
                        self.angular_values <= mid_angle + angle_width - (2 * np.pi))

        if mid_angle - angle_width > 0:
            if (mid_angle - angle_gap) > 0:
                right_angle_mask = (self.angular_values <= mid_angle - angle_gap) & (
                        self.angular_values >= mid_angle - angle_width)
            else:
                raise Exception('(mid_angle - remove_angs) < 0')
        else:
            if (mid_angle - angle_gap) > 0:
                right_angle_mask = (self.angular_values <= mid_angle - angle_gap) | (
                        self.angular_values >= 2 * np.pi + (mid_angle - angle_width))
            else:
                right_angle_mask = (self.angular_values <= mid_angle - angle_gap) | (
                        (self.angular_values >= 2 * np.pi + (mid_angle - angle_width)) & (
                        self.angular_values <= 2 * np.pi + mid_angle - angle_gap))
        # cut masks to same size
        diff = right_angle_mask[right_angle_mask].size - left_angle_mask[left_angle_mask].size
        if diff < 0:
            right_angle_mask[right_angle_mask][:abs(diff)] = False
        elif diff > 0:
            left_angle_mask[left_angle_mask][:abs(diff)] = False
        return left_angle_mask, right_angle_mask

    def __get_radial_mask(self, min_rad, max_rad):
        return (self.radial_values > min_rad) & (self.radial_values < max_rad)

    def __integrate_intensity_for_masks(self, data_idx, left_mask, right_mask, radial_mask):
        radial_masked_data = self.polar_data[data_idx][:, radial_mask]
        # print("left:{0}, right:{1} \n".format(len(radial_masked_data[left_mask]), len(radial_masked_data[
        #                                                                                 right_mask])))
        left_right = np.sum(radial_masked_data[left_mask | right_mask])
        left = np.sum(radial_masked_data[left_mask]) / left_right
        right = np.sum(radial_masked_data[right_mask]) / left_right
        return left, right

    def __plot_polar_data(self, polar_data, label, fig, ax, clev = np.arange(0, 1, .01), cmap='plasma'):
        X_OFFSET = 0

        def add_scale(x):
            rect = x.get_position()
            rect = (rect.xmin - X_OFFSET, rect.ymin + rect.height / 2,  # x, y
                    rect.width, rect.height / 2)
            scale_ax = x.figure.add_axes(rect)
            for loc in ['right', 'top', 'bottom']:
                scale_ax.spines[loc].set_visible(False)
            scale_ax.tick_params(bottom=False, labelbottom=False)
            scale_ax.patch.set_visible(False)
            scale_ax.spines['left'].set_bounds(*x.get_ylim())
            scale_ax.set_yticks(x.get_yticks())
            scale_ax.set_ylim(x.get_rorigin(), x.get_rmax())
            scale_ax.set_ylabel('$r$ / $\\mu m$')

        swapped_data = np.swapaxes(polar_data, 0, 1)

        cs = ax.contourf(self.angular_values, self.radial_values_k, swapped_data[:, ::-1],clev , cmap=cmap)
        cbar = fig.colorbar(cs)
        cbar.set_label(label, rotation=90)
        # label_position = ax.get_rlabel_position()
        # ax.text(np.radians(label_position + 15), ax.get_rmax() / 2., '$|\\vec{k}|$ in $\\mu m^{-1}$',
        #        rotation=label_position, ha='center', va='center', color='silver')
        add_scale(ax)
        ax.set_yticklabels([])
        # ax.set_ylim(0, self.__k_0_NA)
        # ax.legend()
        # fig.tight_layout()

    def __plot_cart_data(self, cart_data, fig, ax, cmap='plasma'):
        cs = ax.imshow(cart_data, cmap=cmap)
        fig.colorbar(cs)

    def plot_cart(self, lambda_4_angle, fig, ax):
        angle, idx = self.__get_lambda_4_index(lambda_4_angle)
        self.__plot_cart_data(self.data[idx], fig, ax)
        ax.set_title('Lambda/4 angle= {0}°'.format(angle))

    def plot_cart_diff(self, lambda_4_angle_1, lambda_4_angle_2, fig, ax):
        angle1, idx1 = self.__get_lambda_4_index(lambda_4_angle_1)
        angle2, idx2 = self.__get_lambda_4_index(lambda_4_angle_2)
        diff = (self.data[idx1] - self.data[idx2]) / (self.data[idx1] + self.data[idx2])
        self.__plot_cart_data(diff, fig, ax, cmap="bwr")
        ax.set_title('Diff, Lambda/4 angle1: {0} - angle2: {1}'.format(angle1, angle2))

    def plot_polar(self, lambda_4_angle, fig, ax):
        angle, idx = self.__get_lambda_4_index(lambda_4_angle)
        label =  '$I_r\\left(|\\vec{r}|, \phi, \\alpha_{\lambda /4} =' + f'{lambda_4_angle}' +'^\circ\\right) /$AU'
        max_value = np.max(self.polar_data[idx])
        self.__plot_polar_data(self.polar_data[idx] / max_value, label, fig, ax)
        # ax.set_title('Lambda/4 angle= {0}°'.format(angle))

    def plot_polar_diff(self, lambda_4_angle_1, lambda_4_angle_2, fig, ax):
        angle1, idx1 = self.__get_lambda_4_index(lambda_4_angle_1)
        angle2, idx2 = self.__get_lambda_4_index(lambda_4_angle_2)
        diff = (self.polar_data[idx1] - self.polar_data[idx2]) / (self.polar_data[idx1] + self.polar_data[idx2])
        label = '$D_r\\left(|\\vec{r}|, \Phi\\right)$'
        self.__plot_polar_data(diff,label, fig, ax, clev=np.arange(-1, 1, .01), cmap="bwr")

        # ax.set_title('Diff, Lambda/4 angle1: {0} - angle2: {1}'.format(angle1, angle2))

    def __calc_radial_profile(self, lambda_4_angle):
        angle, idx = self.__get_lambda_4_index(lambda_4_angle)
        return np.average(self.polar_data[idx], axis=0)


    def plot_angular_profile(self, lambda_4_angle, min_rad, max_rad, fig, ax):
        angle, idx = self.__get_lambda_4_index(lambda_4_angle)
        radial_mask = self.__get_radial_mask(min_rad, max_rad)
        angular_profile = np.average(self.polar_data[idx][:, radial_mask], axis=1)
        ax.plot(self.angular_values, angular_profile, label='$P_r\\left(\\phi, \\alpha_{\\lambda/4}='+f'{lambda_4_angle}'+'^\\circ\\right)$')
        ax.set_yticklabels([])

        #ax.set_title('AngularProfile, Lambda/4 angel{0}'.format(angle))

    def plot_masks(self, mid_angle, angle_width, angle_gap, min_rad, max_rad, fig, ax):
        left_mask, right_mask = self.__get_angle_masks(mid_angle, angle_width, angle_gap)

        left_angles = sorted(self.angular_values[left_mask])
        right_angles = sorted(self.angular_values[right_mask])

        ax.plot(np.full(2,np.min(left_angles)), [min_rad / self.fp_scale, max_rad / self.fp_scale], '-.k', linewidth=1)
        ax.plot(np.full(2,np.max(left_angles)), [min_rad / self.fp_scale, max_rad / self.fp_scale], '-.k', linewidth=1)

        ax.plot(np.full(2,np.min(right_angles)), [min_rad / self.fp_scale, max_rad / self.fp_scale], '-.k', linewidth=1)
        ax.plot(np.full(2,np.max(right_angles)), [min_rad / self.fp_scale, max_rad / self.fp_scale], '-.k', linewidth=1)

        ax.plot(right_angles, np.full(len(right_angles), min_rad / self.fp_scale), '-.k', linewidth=1,
                label='right SPP')
        ax.plot(right_angles, np.full(len(right_angles), max_rad / self.fp_scale), '-.k', linewidth=1)

        ax.plot(left_angles, np.full(len(left_angles), min_rad / self.fp_scale), '-.k', linewidth=1,
                label='left SPP')
        ax.plot(left_angles, np.full(len(left_angles), max_rad / self.fp_scale), '-.k', linewidth=1)

        # ax.plot(np.full(len(self.radial_values_k), mid_angle), self.radial_values_k, linewidth=1)

    def plot_integrated_intensity(self, mid_angle, angle_width, angle_gap, min_rad, max_rad, fig, ax):
        left_mask, right_mask = self.__get_angle_masks(mid_angle, angle_width, angle_gap)
        radial_mask = self.__get_radial_mask(min_rad, max_rad)
        intensities = np.empty((len(self.polar_data), 2))
        for idx, pd in enumerate(self.polar_data):
            intensities[idx] = self.__integrate_intensity_for_masks(idx, left_mask, right_mask, radial_mask)
        # ax.set_xlabel('$\\alpha_{\\lambda/4} / \\mathrm{deg}$')
        # ax.set_ylabel('normed integrated intensity')
        ax.set(xlabel='$\\alpha_{\\lambda/4} / \\mathrm{deg}$',
               ylabel='$P_{r, \\mathrm{relativ}}$',
               xticks=[0, 45, 90, 135, 180, 135, 180, 225, 270, 315, 360])

        sort_index = np.argsort(self.lambda_4_angles)
        ax.plot(self.lambda_4_angles[sort_index], intensities[sort_index, 0], 'g+:', label="$P_{r, \\mathrm{relativ}, \\mathrm{Oben}}(\\alpha_{\\lambda/4})$")
        ax.plot(self.lambda_4_angles[sort_index], intensities[sort_index, 1], 'r+:', label="$P_{r, \\mathrm{relativ}, \\mathrm{Unten}}(\\alpha_{\\lambda/4})$")
        # ax.tick_params(axis='y')

    def plot_polarisation_marks(self, fig, ax):
        circ = np.array([45, 135, 225, 315])
        # lin = np.array([0, 90, 180, 270])
        for c in circ:
            ax.axvline(x=c, linestyle='--', color='k', linewidth='0.5')

    def plot_angular_profile_diff(self, lambda_4_angle_1, lambda_4_angle_2, min_rad, max_rad, fig, ax):
        angle1, idx1 = self.__get_lambda_4_index(lambda_4_angle_1)
        angle2, idx2 = self.__get_lambda_4_index(lambda_4_angle_2)
        radial_mask = self.__get_radial_mask(min_rad, max_rad)
        angular_profile1 = np.average(self.polar_data[idx1][:, radial_mask], axis=1)
        angular_profile2 = np.average(self.polar_data[idx2][:, radial_mask], axis=1)
        diff = (angular_profile1 - angular_profile2) / (angular_profile1 + angular_profile2)
        kernel_size = 1
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved = np.convolve(diff, kernel, mode='same')
        ax.plot(self.angular_values, data_convolved)
        ax.plot(self.angular_values, np.zeros(self.angular_values.shape), '-k')

        def add_scale(x):
            rect = x.get_position()
            rect = (rect.xmin - 0, rect.ymin + rect.height / 2,  # x, y
                    rect.width, rect.height / 2)
            scale_ax = x.figure.add_axes(rect)
            for loc in ['right', 'top', 'bottom']:
                scale_ax.spines[loc].set_visible(False)
            scale_ax.tick_params(bottom=False, labelbottom=False)
            scale_ax.patch.set_visible(False)
            scale_ax.spines['left'].set_bounds(*x.get_ylim())
            scale_ax.set_yticks(x.get_yticks())
            scale_ax.set_ylim(x.get_rorigin(), x.get_rmax())
            scale_ax.set_ylabel('$D_r(\\phi)$')

        #ax.set_ylim(-0.2, 0.2)
        ax.set_rorigin(-0.3)
        add_scale(ax)
        ax.set_yticklabels([])
        # ax.set_yticklabels([])
from __future__ import division
import numpy as np
import scipy.interpolate as interpsci
import copy

class HybridInterpolator:
    def __init__(self, x_array, z_array, val_array):
        self.x_array = x_array
        self.z_array = z_array
        self.val_array = val_array

        self.linear_interpolation = interpsci.LinearNDInterpolator(np.transpose([x_array, z_array]), val_array)
        self.nearest_interpolation = interpsci.NearestNDInterpolator(np.transpose([x_array, z_array]), val_array)

    def __call__(self, x_call_array, z_call_array):
        linear_call = self.linear_interpolation(np.transpose([x_call_array, z_call_array]))
        nearest_call = self.nearest_interpolation(np.transpose([x_call_array, z_call_array]))
        final_call = linear_call
        for j in range(len(final_call)):
            if np.isnan(final_call[j]):
                final_call[j] = nearest_call[j]
        return final_call
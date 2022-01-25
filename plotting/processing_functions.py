from __future__ import division
import numpy as np
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def get_field_values(data_name, folders):
    main_directory = os.getcwd()
    values_array = []
    for folder in folders:
        values = []
        os.chdir(folder)

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(data_name + '{0:06d}.vtu'.format(int(0)))
        reader.Update()
        data = reader.GetOutput()

        f = vtk_to_numpy(data.GetCellData().GetArray(0))

        values.append(f)
        values_array.append(np.asarray(values))

        os.chdir(main_directory)

    return values_array

def get_field_values_step(data_name, folders, step):
    main_directory = os.getcwd()
    values_array = []
    for folder in folders:
        values = []
        os.chdir(folder)

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(data_name + '{0:06d}.vtu'.format(int(step)))
        reader.Update()
        data = reader.GetOutput()

        f = vtk_to_numpy(data.GetCellData().GetArray(0))

        values.append(f)
        values_array.append(np.asarray(values))

        os.chdir(main_directory)

    return values_array

def get_field_values_vel(data_name, folders):
    main_directory = os.getcwd()
    values_array = []
    for folder in folders:
        values = []
        os.chdir(folder)

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(data_name + '{0:06d}.vtu'.format(int(0)))
        reader.Update()
        data = reader.GetOutput()

        f = vtk_to_numpy(data.GetPointData().GetArray(0))

        values.append(f)
        values_array.append(np.asarray(values))

        os.chdir(main_directory)

    return values_array

def get_point_values(data_name, folder, step):
    main_directory = os.getcwd()

    os.chdir(folder)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(data_name + '{0:06d}.vtu'.format(int(step)))
    reader.Update()
    data = reader.GetOutput()

    f = vtk_to_numpy(data.GetPointData().GetArray(0))

    os.chdir(main_directory)

    return [f]

def l2_norm(field):
    return np.sqrt(np.sum(np.square(field)))/np.sqrt((len(field)-1))

def l2_log_norm(field):
    return np.sqrt(np.sum(np.square(np.log10(field))))/np.sqrt((len(field)-1))

def l1_norm(field):
    return np.sum(np.absolute(field))/(len(field)-1)

def field_max(field):
    return np.max(field)

def field_min(field):
    return np.min(field)

def log_field_max(field):
    return np.max(np.log10(field))

def log_field_min(field):
    return np.min(np.log10(field))
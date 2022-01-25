from __future__ import division
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'model')))
import shutil
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import argparse
import paraview.simple as parasim
import seaborn as sns
from processing_functions import *
from geometry_generation import *
matplotlib.rcParams['font.size'] = 6
import scipy.interpolate as interpsci

def plot_length_gl(directory, time_start, time_threshold, axis, bed_interpolator, color_index1, color_index2, labelgl, labelL, ls1='-', ls2='-', eps=1):

    colors = sns.color_palette("colorblind")
    data_name = 'ux_timeseries'

    os.chdir(directory)
    reader_paraview = parasim.PVDReader(FileName=data_name + '.pvd')
    times_imported = reader_paraview.GetPropertyValue('TimestepValues')

    lengths = []
    grounding_line_locations = []
    times = []
    time_index = 'empty'
    for j in range(len(times_imported)):
        time = times_imported[j]
        if time <= time_threshold:
            times.append(time)
        if time >= time_start and time_index == 'empty':
            time_index = j
    print(time_index)

    for iteration in range(len(times)):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(data_name + '{0:06d}.vtu'.format(int(iteration)))
        reader.Update()
        data = reader.GetOutput()
        points = data.GetPoints()
        x = vtk_to_numpy(points.GetData())[:, 0]
        y = vtk_to_numpy(points.GetData())[:, 1]
        locations = np.transpose([x, y])
        sorted_locations = locations[locations[:, 0].argsort()][::-1]
        for j in range(np.shape(sorted_locations)[0]):
            if abs(sorted_locations[j, 1] - bed_interpolator(sorted_locations[j, 0])) <= eps:
                grounding_line_locations.append(sorted_locations[j, 0])
                break
            else:
                pass
        length = data.GetBounds()[1]
        lengths.append(length)
        if len(lengths) != len(grounding_line_locations):
            raise Exception('Failed grounding line detection.')
        if len(lengths) == len(times):
            break

    lengths = np.asarray(lengths)
    grounding_line_locations = np.asarray(grounding_line_locations)

    shifted_lengths = lengths-grounding_line_locations[0]
    shifted_grounding_line = grounding_line_locations - grounding_line_locations[0]
    times_shifted = np.asarray(times) - time_start

    grounded_time = []
    grounded_x = []
    floating_time = []
    floating_x = []
    for j in range(time_index, len(times_shifted)):
        if abs(shifted_grounding_line[j] - shifted_lengths[j]) <= eps:
            pass
        else:
            grounded_time.append(times_shifted[j])
            grounded_x.append(shifted_grounding_line[j])

    t = np.linspace(0, time_threshold, 100)
    d = np.ones(100)*(250+shifted_lengths[0])

    axis.plot(t, d, color='brown', ls='--', zorder=0)

    axis.scatter(times_shifted, shifted_lengths, s=4, color='dimgrey', marker='o', label='Calving Front', zorder=1)
    axis.scatter(grounded_time, grounded_x, s=1, color='k', marker='o', label='Grounding Line (if present)', zorder=2)

    print('Finished ' + str(directory) + '.')



if __name__ == "__main__":
    sns.set(palette='colorblind')
    sns.set(font_scale=0.8)
    colors = sns.color_palette("colorblind")
    sns.set_style("ticks")

    starting_directory = os.getcwd()
    os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests')))
    main_directory = os.getcwd()
    plot_name = "length_matrix_bump"
    time_start = 0.5
    time = 50.5

    geometryX, geometryY, xz_boundary = make_geometry_grounded_latebump(-0.01, 50000, -150, 50, 100, 10, 25, 50250, 50)
    bed_interpolator = interpsci.interp1d(geometryX, geometryY, fill_value='extrapolate')

    directories_base = ['baseline_bump']

    fig = plt.figure(figsize=(7,4.72441/16*9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    axA = [ax1, ax2]

    plot_length_gl(directories_base[0] + '_noad', time_start, time, axA[0], bed_interpolator, 3, 1, 'No Advection w/ Grounded Cliff', 'No Advection w/ Ice Tongue')
    os.chdir(main_directory)
    plot_length_gl(directories_base[0], time_start, time, axA[1], bed_interpolator, 0, 2, 'Advection w/ Grounded Cliff', 'Advection w/ Ice Tongue')
    os.chdir(main_directory)
    os.chdir(starting_directory)

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(handles, labels, loc='lower right',borderaxespad=0.1, ncol=1,  handlelength=0.75)

    ax1.set(xlabel=r"Time (a)", ylabel=r"$\Delta$L (m)", title=r"No Advection")
    ax2.set(xlabel=r"Time (a)", title=r"Advection")

    xlimsA = [axA[j].get_xlim() for j in range(2)]
    ylimsA = [axA[j].get_ylim() for j in range(2)]
    xlim_min = min(np.transpose(xlimsA)[0])
    xlim_max = max(np.transpose(xlimsA)[1])

    ylim_min = min(np.transpose(ylimsA)[0])
    ylim_max = max(np.transpose(ylimsA)[1])

    for j in range(2):
        axA[j].set_xlim([xlim_min, xlim_max])
        axA[j].set_ylim([ylim_min, ylim_max])

    plt.sca(ax2)
    locs = [0, 100, 200, 300, 400, 500, 600, 700]
    plt.yticks(locs, [" "]*len(locs))

    fig_letters = ["a", "b"]
    for j in range(len(axA)):
        axA[j].text(0.025, 0.95, fig_letters[j], transform=axA[j].transAxes, va='top', fontsize=8, weight='bold')
        
    plt.tight_layout(pad=0.1,h_pad=-2.0,w_pad=0.0)

    plt.savefig(plot_name + '.eps')
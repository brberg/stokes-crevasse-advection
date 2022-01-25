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

    axis.scatter(times_shifted[time_index:], shifted_lengths[time_index:], s=4, color='dimgrey', marker='o', label='Calving Front', zorder=1)
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
    plot_name = "length_matrix_slopes"
    time_start = 0.5
    time = 10.5

    bed_interpolators = []
    geometryX, geometryY, xz_boundary = make_geometry_grounded(-0.005, 50000, -150, 50, 100, 10)
    bed_interpolators.append(interpsci.interp1d(geometryX, geometryY, fill_value='extrapolate'))
    geometryX, geometryY, xz_boundary = make_geometry_grounded(-0.02, 50000, -150, 50, 100, 10)
    bed_interpolators.append(interpsci.interp1d(geometryX, geometryY, fill_value='extrapolate'))
    geometryX, geometryY, xz_boundary = make_geometry_retrograde(-0.01, 0.0025, 40000, 50000, -150, 50, 100, 10)
    bed_interpolators.append(interpsci.interp1d(geometryX, geometryY, fill_value='extrapolate'))

    directories = ['baseline_halfslope_noad','baseline_doubleslope_noad','baseline_retrograde_noad']

    labels = ["a", "b", "c"]

    fig = plt.figure(figsize=(7,7/4*3/4*3))
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(323)
    ax3 = fig.add_subplot(325)
    axA = [ax1, ax2, ax3]

    for j in range(len(axA)):
        plot_length_gl(directories[j], time_start, time, axA[j], bed_interpolators[j], 3, 1, 'No Advection w/ Grounded Cliff', 'No Advection w/ Ice Tongue')#, ls1='--')
        os.chdir(main_directory)
        axA[j].text(0.025, 0.9, labels[j], transform=axA[j].transAxes, va='top', fontsize=8, weight='bold')
    os.chdir(starting_directory)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right',borderaxespad=0.1, ncol=1,  handlelength=1.5)

    ax1.set(ylabel=r"$\Delta$L (m)", title="No Advection")
    ax2.set(ylabel=r"$\Delta$L (m)")
    ax3.set(ylabel=r"$\Delta$L (m)")

    ax3.set(xlabel=r"Time (a)")

    xlimsA = [axA[j].get_xlim() for j in range(3)]
    ylimsA = [axA[j].get_ylim() for j in range(3)]
    xlim_minA = min(np.transpose(xlimsA)[0])
    xlim_maxA = max(np.transpose(xlimsA)[1])

    #######################################################
    os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests')))
    main_directory = os.getcwd()

    directories = ['baseline_halfslope_ad','baseline_doubleslope_ad','baseline_retrograde_ad']
    plot_names = ['half slope', 'double slope', 'retrograde slope']

    labels = ["d", "e", "f"]

    ax5 = fig.add_subplot(322)
    ax6 = fig.add_subplot(324)
    ax7 = fig.add_subplot(326)
    axB = [ax5, ax6, ax7]

    for j in range(len(axB)):
        plot_length_gl(directories[j], time_start, time, axB[j], bed_interpolators[j], 3, 1, 'No Advection w/ Grounded Cliff', 'No Advection w/ Ice Tongue')
        os.chdir(main_directory)
        axB[j].text(0.025, 0.9, labels[j], transform=axB[j].transAxes, va='top', fontsize=8, weight='bold')
        axB[j].text(1.05, 0.5, plot_names[j], transform=axB[j].transAxes, va='center', fontsize=8, rotation='vertical')
    os.chdir(starting_directory)

    ax5.set(title="Advection")

    ax7.set(xlabel=r"Time (a)")

    xlimsB = [axB[j].get_xlim() for j in range(3)]
    ylimsB = [axB[j].get_ylim() for j in range(3)]
    ylimsmin = [min(ylimsA[j][0], ylimsB[j][0]) for j in range(3)]
    ylimsmax = [max(ylimsA[j][1], ylimsB[j][1]) for j in range(3)]

    ymintop = ylimsmin[0]
    ymaxtop = ylimsmax[0]
    yminbot = min(ylimsmin[1], ylimsmin[2])
    ymaxbot = max(ylimsmax[1], ylimsmax[2])

    ylimsminfinal = [ymintop, yminbot, yminbot]
    ylimsmaxfinal = [ymaxtop, ymaxbot, ymaxbot]

    xlim_minB = min(np.transpose(xlimsB)[0])
    xlim_maxB = max(np.transpose(xlimsB)[1])
    xlim_min = min(xlim_minA, xlim_minB)
    xlim_max = max(xlim_maxA, xlim_maxB)
    for j in range(3):
        axA[j].set_xlim([xlim_min, xlim_max])
        axB[j].set_xlim([xlim_min, xlim_max])
        axA[j].set_ylim([ylimsminfinal[j], ylimsmaxfinal[j]])
        axB[j].set_ylim([ylimsminfinal[j], ylimsmaxfinal[j]])

    plt.sca(ax1)
    xlims = plt.xticks()
    locs = xlims[0][1:-1]
    labels = []
    for j in range(len(locs)):
        labels.append('%.0f'%(locs[j]))
    for axis in [ax1, ax2, ax5, ax6]:
        plt.sca(axis)
        plt.xticks(locs, [" "]*len(locs))
    for axis in [ax3, ax7]:
        plt.sca(axis)
        plt.xticks(locs, labels)

    plt.sca(ax5)
    ylims = plt.yticks()
    locs = ylims[0][0:-1]
    labels = []
    for j in range(len(locs)):
        labels.append('%.0f'%(locs[j]))
    plt.sca(ax5)
    plt.yticks(locs, [" "]*len(locs))
    plt.sca(ax1)
    plt.yticks(locs, labels)

    plt.sca(ax6)
    ylims = plt.yticks()
    locs = ylims[0][0:-1]
    labels = []
    for j in range(len(locs)):
        labels.append('%.0f'%(locs[j]))
    plt.sca(ax6)
    plt.yticks(locs, [" "]*len(locs))
    plt.sca(ax2)
    plt.yticks(locs, labels)

    plt.sca(ax7)
    ylims = plt.yticks()
    locs = ylims[0][0:-1]
    labels = []
    for j in range(len(locs)):
        labels.append('%.0f'%(locs[j]))
    plt.sca(ax7)
    plt.yticks(locs, [" "]*len(locs))
    plt.sca(ax3)
    plt.yticks(locs, labels)

    plt.tight_layout(pad=0.1,h_pad=-1.0,w_pad=0.0)

    plt.savefig(plot_name + '.eps')
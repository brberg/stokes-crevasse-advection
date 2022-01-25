from __future__ import division
import numpy as np
import sys
import os
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
import multiprocessing as mp
import copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'model')))
from geometry_generation import *
matplotlib.rcParams['font.size'] = 6
import scipy.interpolate as interpsci
import seaborn as sns

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def makeImages(snapshots, geometryX, geometryY, data_names, folder_name, time):

    root_directory = os.getcwd()

    fig = plt.figure(figsize=(7/2,4.72441/4*3))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    axes = [ax1, ax2]

    for n in range(len(snapshots)):

        i = snapshots[n]
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(data_names[n] + '{0:06d}.vtu'.format(i))
        reader.Update()
        data = reader.GetOutput()
        points = data.GetPoints()
        npts = points.GetNumberOfPoints()
        x = vtk_to_numpy(points.GetData())[:, 0]
        y = vtk_to_numpy(points.GetData())[:, 1]
        f = vtk_to_numpy(data.GetPointData().GetArray(0))
        triangles = vtk_to_numpy(data.GetCells().GetData())
        ntri = triangles.size//4
        tri = np.take(triangles,[m for m in range(triangles.size) if m%4 != 0]).reshape(ntri, 3)

        waterX = np.linspace(0, 60000, 100)
        waterY = np.zeros(100)

        levels = np.linspace(0, 1.0, 100, endpoint=True)

        cmap_new = truncate_colormap(plt.get_cmap("BuPu"), 0.25, 1.0)

        bed_interpolator = interpsci.interp1d(geometryX, geometryY, fill_value='extrapolate')
        geometryX = np.linspace(0, 60000, 1000)
        geometryY = bed_interpolator(geometryX)
        axes[n].fill_between(waterX, -200, 0, color='#94aec4ff', zorder=-21)
        axes[n].fill_between(geometryX, -200, geometryY, color='#c69d6eff',zorder=-18)

        cnt = axes[n].tricontourf(x, y, tri, f*100, 100, cmap=cmap_new, levels=levels, extend='both', zorder=-20)
        for c in cnt.collections:
            c.set_edgecolor("face")
        if n==1:
            axes[n].axvline(50983.2, 0.19, 1, color='white', ls='--')
        axes[n].set_rasterization_zorder(-10)

        print("Processed file number " + str(i) + ".")

        axes[n].set_xlim([50400,51600])
        axes[n].set_ylim([-200,50])

        axes[n].set_xlabel('Distance (km)')
        axes[n].set_ylabel('Height (m)')

        plt.sca(axes[n])
        plt.xticks([50500, 51000, 51500], ["50.5", "51.0", "51.5"])

        os.chdir(root_directory)

    plt.sca(axes[0])
    plt.xticks([50500, 51000, 51500], [])
    plt.tight_layout(pad=0.1,h_pad=-1.0,w_pad=0.0)
    axes[0].set_xlabel('')

    labels = ["a", "b"]

    for n in range(len(axes)):
        axes[n].text(-0.175, 1.025, labels[n], transform=axes[n].transAxes, va='top', fontsize=8, weight='bold')

    fig.savefig(folder_name + "/" + "thumbnails_calving.eps", transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    starting_directory = os.getcwd()
    os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests')))
    main_directory = os.getcwd()

    sns.set(palette='colorblind')
    sns.set(font_scale=0.8)
    sns.set_style(style='ticks')

    geometryX, geometryY, xz_boundary = make_geometry_grounded(-0.01, 50000, -150, 50, 100, 10)
    time = 5.81
    directories = ['slippery_noad', 'slippery']
    dataName = 'width_timeseries'

    snapshots = []
    data_names = []

    for directory in directories:
        data_names.append(os.path.join(directory, dataName))
        if time == int(0):
            snapshots.append(int(0))
        else:
            os.chdir(directory)
            reader_paraview = parasim.PVDReader(FileName=dataName + '.pvd')
            times_imported = reader_paraview.GetPropertyValue('TimestepValues')
            times_temp = 0.0
            for k in range(len(times_imported)):
                if times_imported[k] >= time and times_temp <= time:
                    snapshots.append(int(k))
                    break
                else:
                    times_temp = times_imported[k]
        os.chdir(main_directory)

    makeImages(snapshots, geometryX, geometryY, data_names, starting_directory, time)
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

        print(data_names)
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

        readerN = vtk.vtkXMLUnstructuredGridReader()
        readerN.SetFileName(data_names[n+1] + '{0:06d}.vtu'.format(i))
        readerN.Update()
        dataN = readerN.GetOutput()
        pointsN = dataN.GetPoints()
        nptsN = pointsN.GetNumberOfPoints()
        xN = vtk_to_numpy(pointsN.GetData())[:, 0]
        yN = vtk_to_numpy(pointsN.GetData())[:, 1]
        fN = vtk_to_numpy(dataN.GetPointData().GetArray(0))
        trianglesN = vtk_to_numpy(dataN.GetCells().GetData())
        ntriN = trianglesN.size//4
        triN = np.take(trianglesN,[m for m in range(trianglesN.size) if m%4 != 0]).reshape(ntriN, 3)

        waterX = np.linspace(0, 60000, 100)
        waterY = np.zeros(100)

        levels = np.linspace(0, 1.0, 100, endpoint=True)

        cmap_new = truncate_colormap(plt.get_cmap("BuPu"), 0.25, 1.0)

        center = 0
        rangeCB = 10000
        levelsN = np.linspace((center-rangeCB)/1000, (center+rangeCB)/1000, 100, endpoint=True)
        cmap_newN = plt.get_cmap('seismic')
        cmap_newN2 = truncate_colormap(plt.get_cmap("binary"), 0.99, 1.0)

        bed_interpolator = interpsci.interp1d(geometryX, geometryY, fill_value='extrapolate')
        geometryX = np.linspace(0, 60000, 1000)
        geometryY = bed_interpolator(geometryX)
        axes[n].fill_between(waterX, -200, 0, color='#94aec4ff', zorder=-21)
        axes[n].fill_between(geometryX, -200, geometryY, color='#c69d6eff',zorder=-18)
        axes[n+1].fill_between(waterX, -200, 0, color='#94aec4ff', zorder=-21)
        axes[n+1].fill_between(geometryX, -200, geometryY, color='#c69d6eff',zorder=-18)

        cnt = axes[n].tricontourf(x, y, tri, f*100, 100, cmap=cmap_new, levels=levels, extend='both', zorder=-20)
        cnt2 = axes[n+1].tricontourf(xN, yN, triN, fN/1000, 100, cmap=cmap_newN, levels=levelsN, extend='both', zorder=-20)
        axes[n+1].tricontour(xN, yN, triN, fN/1000, levels=[center/1000], cmap=cmap_newN2, linestyles='dashed')
        for c in cnt.collections:
            c.set_edgecolor("face")
        for c in cnt2.collections:
            c.set_edgecolor("face")
        axes[n].set_rasterization_zorder(-10)
        axes[n+1].set_rasterization_zorder(-10)

        print("Processed file number " + str(i) + ".")

        axes[n].set_xlim([49800,50000])
        axes[n].set_ylim([-200,50])
        axes[n+1].set_xlim([49800,50000])
        axes[n+1].set_ylim([-200,50])

        axes[n].set_xlabel('Distance (km)')
        axes[n].set_ylabel('Height (m)')
        axes[n+1].set_xlabel('Distance (km)')
        axes[n+1].set_ylabel('Height (m)')

        plt.sca(axes[n])
        plt.xticks([49800, 49900, 50000], ["49.8", "49.9", "50.0"])
        plt.yticks([-200,-150,-100,-50,0,50], ["-200", "-150" ,"-100", "-50", "0", "50"])
        plt.sca(axes[n+1])
        plt.xticks([49800, 49900, 50000], ["49.8", "49.9", "50.0"])
        plt.yticks([-200,-150,-100,-50,0,50], ["-200", "-150" ,"-100", "-50", "0", "50"])

        os.chdir(root_directory)

    plt.subplots_adjust(bottom=0.3, left=0.17, right=0.965, top=0.985)
    cb_ax = fig.add_axes([0.16, 0.12, 0.8, 0.02])
    cbar = plt.colorbar(cnt2, orientation="horizontal", ticks=[(center-rangeCB)/1000, center/1000, (center+rangeCB)/1000], cax=cb_ax)
    cbar.set_label(label="Nye Stress (kPa)", size=8)
    cbar.ax.tick_params(labelsize=8) 


    plt.sca(axes[0])
    plt.xticks([49800, 49900, 50000], [])
    axes[0].set_xlabel('')

    labels = ["a", "b"]

    for n in range(len(axes)):
        axes[n].text(-0.175, 1.025, labels[n], transform=axes[n].transAxes, va='top', fontsize=8, weight='bold')

    fig.savefig(folder_name + "/" + "thumbnails_nyewidth.eps")
    plt.close(fig)


if __name__ == "__main__":
    starting_directory = os.getcwd()
    os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests')))
    main_directory = os.getcwd()

    sns.set(palette='colorblind')
    sns.set(font_scale=0.8)
    sns.set_style(style='ticks')

    geometryX, geometryY, xz_boundary = make_geometry_grounded(-0.01, 50000, -150, 50, 100, 10)
    time = 2.00
    directories = ['warm']
    dataName = 'width_timeseries'

    snapshots = []
    data_names = []

    for directory in directories:
        data_names.append(os.path.join(directory, dataName))
        data_names.append(os.path.join(directory, 'nyestress_timeseries'))
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
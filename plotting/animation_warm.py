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

def makeImages(iter, snapshots, geometryX, geometryY, data_names, folder_name, time):

    root_directory = os.getcwd()

    fig = plt.figure(figsize=(7,4.72441/4*3))

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
        axes[n].set_rasterization_zorder(-10)

        print("Processed file number " + str(i) + ".")

        axes[n].set_xlim([49500,51000])
        axes[n].set_ylim([-200,50])

        axes[n].set_xlabel('Distance (km)')
        axes[n].set_ylabel('Height (m)')

        plt.sca(axes[n])
        plt.xticks([49500, 50000, 50500, 51000], ["49.5", "50.0", "50.5", "51.0"])

        os.chdir(root_directory)

    plt.sca(axes[0])
    plt.xticks([49500, 50000, 50500, 51000], [])
    plt.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.9)
    axes[0].set_xlabel('')

    labels = ["a", "b"]

    for n in range(len(axes)):
        axes[n].text(-0.175, 1.025, labels[n], transform=axes[n].transAxes, va='top', fontsize=8, weight='bold')

    ax1.text(-0.15, 0.5, 'No Advection', transform=ax1.transAxes, va='center', fontsize=12, rotation='vertical')
    ax2.text(-0.15, 0.5, 'Advection', transform=ax2.transAxes, va='center', fontsize=12, rotation='vertical')
    fig.suptitle("t = %.1f years" % (time-0.5))

    fig.savefig(folder_name + "/" + "animation_warm" + str(iter) + ".png", transparent=False, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    starting_directory = os.getcwd()
    os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests')))
    main_directory = os.getcwd()

    sns.set(palette='colorblind')
    sns.set(font_scale=0.8)
    sns.set_style(style='ticks')

    geometryX, geometryY, xz_boundary = make_geometry_grounded(-0.01, 50000, -150, 50, 100, 10)

    #time = 5.81
    max_time = 15.0
    time_step = 1/12
    directories = ['warm_noad', 'warm']
    dataName = 'width_timeseries'
    folder_name = "animation_warm"
    folder_directory = os.path.join(starting_directory, folder_name)

    if os.path.isdir(folder_directory):
        shutil.rmtree(folder_directory)
    os.makedirs(folder_directory)

    times = np.linspace(0+0.5, max_time+0.5, int(max_time/time_step))

    snapshots = [[], []]
    data_names = []

    for count, directory in enumerate(directories):
        data_names.append(os.path.join(directory, dataName))
        os.chdir(directory)
        for time in times:
            if time == int(0):
                snapshots[count].append(int(0))
            else:
                reader_paraview = parasim.PVDReader(FileName=dataName + '.pvd')
                times_imported = reader_paraview.GetPropertyValue('TimestepValues')
                times_temp = 0.0
                for k in range(len(times_imported)):
                    if times_imported[k] >= time and times_temp <= time:
                        snapshots[count].append(int(k))
                        break
                    else:
                        times_temp = times_imported[k]
        os.chdir(main_directory)

    snapshots = np.transpose(snapshots)

    for j in range(len(times)):
        makeImages(j, snapshots[j], geometryX, geometryY, data_names, folder_directory, times[j])

    os.chdir(folder_directory)
    Writer = animation.writers['ffmpeg']
    writer = Writer(codec="libx264", extra_args=['-pix_fmt', 'yuv420p'], bitrate=-1)

    fig = plt.figure()
    ims = []
    for i in range(len(times)):
        imraw = mpimg.imread("animation_warm" + str(i) + ".png")
        im = plt.imshow(imraw, animated=True)
        plt.axis('off')
        ims.append([im])

        print("Loaded image number " + str(i) + ".")

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True) #interval=100
    print("Animation generated.")
    ani.save("animation_warm.mp4", writer=writer, dpi=300) #dpi=500
    print("Animation saved.")
 
    print("Process complete.")
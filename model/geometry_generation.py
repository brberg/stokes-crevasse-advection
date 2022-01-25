from __future__ import division
import numpy as np
import os
from pathlib import Path
import copy

def make_geometry_verification(N=10):

    x_points = np.linspace(0, 1, N)
    z_points = np.linspace(0, 1, N)[1:-1]

    z_bottom = np.zeros(N)
    z_top = np.ones(N)

    xz_top = [x_points[::-1], z_top[::-1]]
    xz_bottom = [x_points, z_bottom]
    xz_left = [np.zeros(N-2), z_points[::-1]]
    xz_right = [np.ones(N-2), z_points]
    xz_sides = [xz_left, xz_bottom, xz_right, xz_top]

    x_boundary = []
    z_boundary = []
    for xz_side in xz_sides:
        for j in range(len(xz_side[0])):
            x_boundary.append(xz_side[0][j])
            z_boundary.append(xz_side[1][j])

    xz_boundary = [x_boundary, z_boundary]

    boundary_path = str(Path(os.getcwd()) / "boundary-points-verification.csv")
    np.savetxt(boundary_path, xz_boundary, delimiter=',')

    return xz_boundary

def make_geometry_diagnostic_shelf(N=1000):

    x_points = np.linspace(0, 10000, N)
    m = 0.0
    x1, y1 = 10000, -90
    x2, y2 = 10000, 10

    z_bottom = y1+m*(x_points-x1)
    z_top = y2+m*(x_points-x2)

    xz_top = [x_points[::-1], z_top[::-1]]
    xz_bottom = [x_points, z_bottom]
    xz_left = [np.zeros(N//100), np.linspace(z_bottom[0]+(y2-y1)/(N//100), z_top[0], num=N//100, endpoint = False)[::-1]]
    xz_right = [x1*np.ones(N//100), np.linspace(z_bottom[-1]+(y2-y1)/(N//100), z_top[-1], num=N//100, endpoint = False)]
    xz_sides = [xz_left, xz_bottom, xz_right, xz_top]

    x_boundary = []
    z_boundary = []
    for xz_side in xz_sides:
        for j in range(len(xz_side[0])):
            x_boundary.append(xz_side[0][j])
            z_boundary.append(xz_side[1][j])

    xz_boundary = [x_boundary, z_boundary]

    boundary_path = str(Path(os.getcwd()) / "boundary-points-diagnostic-shelf.csv")
    np.savetxt(boundary_path, xz_boundary, delimiter=',')

    return xz_boundary

def make_geometry_flat_pig(N=1000):

    x_points = np.linspace(0, 50000, N)
    m = 0.0
    x1, y1 = 50000, -360
    x2, y2 = 50000, 40

    z_bottom = y1+m*(x_points-x1)
    z_top = y2+m*(x_points-x2)

    xz_top = [x_points[::-1], z_top[::-1]]
    xz_bottom = [x_points, z_bottom]
    xz_left = [np.zeros(N//100), np.linspace(z_bottom[0]+(y2-y1)/(N//100), z_top[0], num=N//100, endpoint = False)[::-1]]
    xz_right = [x1*np.ones(N//100), np.linspace(z_bottom[-1]+(y2-y1)/(N//100), z_top[-1], num=N//100, endpoint = False)]
    xz_sides = [xz_left, xz_bottom, xz_right, xz_top]

    x_boundary = []
    z_boundary = []
    for xz_side in xz_sides:
        for j in range(len(xz_side[0])):
            x_boundary.append(xz_side[0][j])
            z_boundary.append(xz_side[1][j])

    xz_boundary = [x_boundary, z_boundary]

    boundary_path = str(Path(os.getcwd()) / "boundary-points-flat-pig.csv")
    np.savetxt(boundary_path, xz_boundary, delimiter=',')

    return xz_boundary

def make_geometry_gaussian_pig(N=1000, N2=40):

    a = 100
    fwhm = 50
    c = fwhm/(2*np.sqrt(2*np.log(2)))
    b = 10000

    x_points_1 = np.linspace(0, b-100, int(N*(b-100-0)/50000), endpoint=False)
    x_points_gaussian = np.linspace(b-100, b+100, N2, endpoint=False)
    x_points_3 = np.linspace(b+100, 50000, int(N*(50000-(b+100))/50000))
    x_points = np.concatenate((x_points_1, x_points_gaussian, x_points_3))
    m = 0.0
    x1, y1 = 50000, -360
    x2, y2 = 50000, 40


    z_bottom = y1+m*(x_points-x1) + a*np.exp((-(x_points-b)**2)/(2*c**2))
    z_top = y2+m*(x_points-x2) 

    xz_top = [x_points[::-1], z_top[::-1]]
    xz_bottom = [x_points, z_bottom]
    xz_left = [np.zeros(N//100), np.linspace(z_bottom[0]+(y2-y1)/(N//100), z_top[0], num=N//100, endpoint = False)[::-1]]
    xz_right = [x1*np.ones(N//100), np.linspace(z_bottom[-1]+(y2-y1)/(N//100), z_top[-1], num=N//100, endpoint = False)]
    xz_sides = [xz_left, xz_bottom, xz_right, xz_top]

    x_boundary = []
    z_boundary = []
    for xz_side in xz_sides:
        for j in range(len(xz_side[0])):
            x_boundary.append(xz_side[0][j])
            z_boundary.append(xz_side[1][j])

    xz_boundary = [x_boundary, z_boundary]

    boundary_path = str(Path(os.getcwd()) / "boundary-points-flat-pig.csv")
    np.savetxt(boundary_path, xz_boundary, delimiter=',')

    return xz_boundary

def make_geometry_diagnostic_grounded(N=1000):

    x_points = np.linspace(0, 10000, N)
    m = 0.0
    x1, y1 = 10000, -50
    x2, y2 = 10000, 50

    z_bottom = y1+m*(x_points-x1)
    z_top = y2+m*(x_points-x2)

    xz_top = [x_points[::-1], z_top[::-1]]
    xz_bottom = [x_points, z_bottom]
    xz_left = [np.zeros(N//100), np.linspace(z_bottom[0]+(y2-y1)/(N//100), z_top[0], num=N//100, endpoint = False)[::-1]]
    xz_right = [x1*np.ones(N//100), np.linspace(z_bottom[-1]+(y2-y1)/(N//100), z_top[-1], num=N//100, endpoint = False)]
    xz_sides = [xz_left, xz_bottom, xz_right, xz_top]

    x_boundary = []
    z_boundary = []
    for xz_side in xz_sides:
        for j in range(len(xz_side[0])):
            x_boundary.append(xz_side[0][j])
            z_boundary.append(xz_side[1][j])

    xz_boundary = [x_boundary, z_boundary]

    boundary_path = str(Path(os.getcwd()) / "boundary-points-diagnostic-shelf.csv")
    np.savetxt(boundary_path, xz_boundary, delimiter=',')

    return xz_boundary

def make_bed_diagnostic_grounded(N=1000):
    bed_points_x = np.linspace(0, 10000, N)
    bed_points_z = -50*np.ones(N)

    return bed_points_x, bed_points_z

def make_geometry_grounded(m, L, y1, y2, N, N2):

    x_points = np.linspace(0, L, N)
    x1, y1 = L, y1
    x2, y2 = L, y2

    z_bottom = y1+m*(x_points-x1)
    z_top = y2+m*(x_points-x2)

    xz_top = [x_points[::-1], z_top[::-1]]
    xz_bottom = [x_points, z_bottom]
    xz_left = [np.zeros(N2), np.linspace(z_bottom[0]+(y2-y1)/N2, z_top[0], num=N2, endpoint = False)[::-1]]
    xz_right = [x1*np.ones(N2), np.linspace(z_bottom[-1]+(y2-y1)/(N2), z_top[-1], num=N2, endpoint = False)]
    xz_sides = [xz_left, xz_bottom, xz_right, xz_top]

    x_boundary = []
    z_boundary = []
    for xz_side in xz_sides:
        for j in range(len(xz_side[0])):
            x_boundary.append(xz_side[0][j])
            z_boundary.append(xz_side[1][j])

    xz_boundary = [x_boundary, z_boundary]

    boundary_path = str(Path(os.getcwd()) / "boundary-points-grounded.csv")
    np.savetxt(boundary_path, xz_boundary, delimiter=',')

    return xz_bottom[0], xz_bottom[1], xz_boundary

def make_geometry_grounded_latebump(m, L, y1, y2, N, N2, a, b, c):

    x_points_A = np.linspace(0, b-5*c, N//2, endpoint=False)
    x_points_B = np.linspace(b-5*c, b+5*c, N, endpoint=False)
    x_points_C = np.linspace(b+5*c, L, N//2)
    x_points = np.append(np.append(x_points_A, x_points_B), x_points_C)
    x1, y1 = L, y1
    x2, y2 = L, y2

    bump = a*np.exp(-np.power(x_points-b, 2)/(2*c**2))
    z_bottom = y1+m*(x_points-x1) + bump
    z_top = y2+m*(x_points-x2)

    xz_top = [x_points[::-1], z_top[::-1]]
    xz_bottom = [x_points, z_bottom]
    xz_left = [np.zeros(N2), np.linspace(z_bottom[0]+(y2-y1)/N2, z_top[0], num=N2, endpoint = False)[::-1]]
    xz_right = [x1*np.ones(N2), np.linspace(z_bottom[-1]+(y2-y1)/(N2), z_top[-1], num=N2, endpoint = False)]
    xz_sides = [xz_left, xz_bottom, xz_right, xz_top]

    x_boundary = []
    z_boundary = []
    for xz_side in xz_sides:
        for j in range(len(xz_side[0])):
            x_boundary.append(xz_side[0][j])
            z_boundary.append(xz_side[1][j])

    xz_boundary = [x_boundary, z_boundary]

    boundary_path = str(Path(os.getcwd()) / "boundary-points-grounded.csv")
    np.savetxt(boundary_path, xz_boundary, delimiter=',')

    return xz_bottom[0], xz_bottom[1], xz_boundary

def make_geometry_retrograde(m1, m2, La, L, y1, y2, N, N2):

    x_points = np.linspace(0, L, N)
    x1, y1 = L, y1
    x2, y2 = L, y2

    y1i = y1 - m2*(L-La)-m1*(La)
    y2i = y2 - m2*(L-La)-m1*(La)
    y1h = y1i + m1*(La)
    y2h = y2i + m1*(La)

    z_bottom = (y1i+m1*x_points)*(x_points <= La) + (y1h+m2*(x_points-La))*(x_points > La)
    z_top = (y2i+m1*x_points)*(x_points <= La) + (y2h+m2*(x_points-La))*(x_points > La)

    xz_top = [x_points[::-1], z_top[::-1]]
    xz_bottom = [x_points, z_bottom]
    xz_left = [np.zeros(N2), np.linspace(z_bottom[0]+(y2-y1)/N2, z_top[0], num=N2, endpoint = False)[::-1]]
    xz_right = [x1*np.ones(N2), np.linspace(z_bottom[-1]+(y2-y1)/(N2), z_top[-1], num=N2, endpoint = False)]
    xz_sides = [xz_left, xz_bottom, xz_right, xz_top]

    x_boundary = []
    z_boundary = []
    for xz_side in xz_sides:
        for j in range(len(xz_side[0])):
            x_boundary.append(xz_side[0][j])
            z_boundary.append(xz_side[1][j])

    xz_boundary = [x_boundary, z_boundary]

    boundary_path = str(Path(os.getcwd()) / "boundary-points-retrograde.csv")
    np.savetxt(boundary_path, xz_boundary, delimiter=',')

    return xz_bottom[0], xz_bottom[1], xz_boundary
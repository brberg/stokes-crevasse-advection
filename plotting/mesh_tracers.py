from __future__ import division
#Add path with model classes and import
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'model')))
from geometry_generation import *
from meshing import *
from solver import *
from saving import *
import pyvista as pv
import matplotlib
import scipy.interpolate as interpsci

starting_directory = os.getcwd()

#Set Directory
sim_name = sys.argv[0][0:-3]
parameter_file_name = sim_name + '.py'
setup_folder(sim_name, parameter_file_name, extra_files=['baseline_ss_mesh.xml'])

#Make geometry .csv files
bed_points_x, bed_points_z, xz_boundary = make_geometry_grounded(-0.01, 50000, -150, 50, 100, 10)

#Make mesh
FenicsMesh = FenicsMesh(cell_size=100, hires_cell_size=10)
FenicsMesh.set_bed_location(bed_points_x, bed_points_z)
FenicsMesh.set_smb(np.linspace(0, 50000, 100), 0.25*np.ones(100))
FenicsMesh.load_mesh('baseline_ss_mesh.xml')

seconds_per_year = 3.154E7
SolveClass = SimulationSolver(FenicsMesh, friction_coefficient=7.6E6/(seconds_per_year**(1/3)), solver_tolerance=1E-4, viscosity_type = 'glen', creep_parameter=3.5E-25*seconds_per_year, save_all=False, save_mesh=True, advect=True, calving_type='particle', calving_start=0.5, yield_width=0.0099)

os.chdir(starting_directory)

p = plot(FenicsMesh.mesh, wireframe=True, color='k', lw=0.5)
plt.xlim([49700, 50025])
plt.ylim([-175, 25])

waterX = np.linspace(50000, 50025, 100)

plt.fill_between(waterX, -200, 0, color='#94aec4ff', zorder=-21)
plt.fill_between(bed_points_x, -200, bed_points_z, color='#c69d6eff', zorder=-18)

r = SolveClass.Tracers.particles.return_property(SolveClass.Tracers.FenicsMesh.mesh, 0)
rT = np.transpose(r)

x = rT[0]
z = rT[1]

bed_interpolator = interpsci.interp1d(bed_points_x, bed_points_z, fill_value='extrapolate')


[x_g, z_g] = SolveClass.Tracers.FenicsMesh.get_xz_boundary_points()
glacier_bot_x = []
glacier_bot_z = []
for j in range(len(x_g)):
    if z_g[j] <= 0:
        glacier_bot_x.append(x_g[j])
        glacier_bot_z.append(z_g[j])

bed_points_x_truncated = []
bed_points_z_truncated = []
glacier_bot_z_truncated = []

for j in range(len(glacier_bot_x)):
    if abs(glacier_bot_z[j]-bed_interpolator(glacier_bot_x[j])) >= 0.1:
        bed_points_x_truncated.append(glacier_bot_x[j])
        bed_points_z_truncated.append(bed_interpolator(glacier_bot_x[j]))
        glacier_bot_z_truncated.append(glacier_bot_z[j])

plt.fill_between(bed_points_x_truncated, bed_points_z_truncated, glacier_bot_z_truncated, color='#94aec4ff', zorder=-21)

plt.scatter(x, z, color='gray',s=0.02, marker=",")

plt.axis('off')

plt.savefig('mesh_tracers.eps')
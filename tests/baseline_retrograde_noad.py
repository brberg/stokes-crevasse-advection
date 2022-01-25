from __future__ import division
#Add path with model classes and import
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'model')))
from geometry_generation import *
from meshing import *
from solver import *
from saving import *

#Set Directory
sim_name = sys.argv[0][0:-3]
parameter_file_name = sim_name + '.py'
setup_folder(sim_name, parameter_file_name, extra_files=['retrograde_ss_mesh.xml'])

#Make geometry .csv files
bed_points_x, bed_points_z, xz_boundary = make_geometry_retrograde(-0.01, 0.0025, 40000, 50000, -150, 50, 100, 10)

#Make mesh
FenicsMesh = FenicsMesh(cell_size=100, hires_cell_size=10)
FenicsMesh.set_bed_location(bed_points_x, bed_points_z)
FenicsMesh.set_smb(np.linspace(0, 50000, 100), 0.25*np.ones(100))
FenicsMesh.load_mesh('retrograde_ss_mesh.xml')

seconds_per_year = 3.154E7
SolveClass = SimulationSolver(FenicsMesh, friction_coefficient=7.6E6/(seconds_per_year**(1/3)), solver_tolerance=1E-4, viscosity_type = 'glen', creep_parameter=3.5E-25*seconds_per_year, save_all=False, save_mesh=True, advect=False, calving_type='particle', calving_start=0.5, yield_width=0.0099)

SolveClass.run_model(duration=10.5, dt=3/(365))
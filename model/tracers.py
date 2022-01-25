from __future__ import division
import numpy as np
import scipy.interpolate as interpsci
import matplotlib.path as mpltPath
from dolfin import  *
from hybrid_interpolator import *
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

from exceptions import *

from leopart import (
    particles,
    l2projection,
    RandomCell,
    AddDelete,
    advect_particles,
)

def strain_rate(velocity):
    return 0.5*(nabla_grad(velocity) + nabla_grad(velocity).T)

def solve_zero(f,x):
    s = np.sign(f)
    z1 = np.where(s == 0)[0]
    if len(z1) > 0:
        z1 = z1[0]
    s = s[0:-1] + s[1:]
    z2 = np.where(s == 0)[0]
    if len(z2) > 0:
        z2 = z2[0]
    if z1 and z2:
        return min(z1, z2)
    elif z1:
        return z1
    else:
        return z2

def interpolate_zero(f,x,z):
    if z:
        if z == range(len(f))[-1]:
            return x[z]
        else:
            m = (f[z+1] - f[z]) / (x[z+1] - x[z])
            if m == 0.0:
                return x[z]
        return x[z] - f[z]/m
    else:
        return False

def intersection_zero(x, y1, y2):
    f = y1 - y2
    z = solve_zero(f, x)
    ans = interpolate_zero(f, x, z)
    return ans

class Tracers:
    def __init__(self, FenicsMesh, rho_i, rho_w, g, advect, calving_type, yield_strength, yield_width, opening_width, smoothing_width, tracers_per_cell, spacing_calving_check, boundary_tolerance, viscosity_type, creep_parameter, newton_viscosity, epsilon, calved_threshold=0.99):
        self.FenicsMesh = FenicsMesh
        self.tracers_per_cell = tracers_per_cell
        self.spacing_calving_check = spacing_calving_check

        self.rho_i = rho_i
        self.rho_w = rho_w
        self.g = g

        self.boundary_tolerance = boundary_tolerance

        self.calving_type = calving_type	
        self.calving_functions = {
            'particle' : self.find_calving_location_particle,
        }
        self.update_functions = {	
            'particle' : self.update_particle_tracers,	
        }

        self.advect = advect
        self.yield_strength = yield_strength
        self.yield_width = yield_width
        self.opening_width = opening_width
        self.smoothing_width = smoothing_width

        gen = RandomCell(FenicsMesh.mesh)
        x = gen.generate(tracers_per_cell)
        f = np.zeros(np.shape(x)[0])
        self.particles = particles(x, [f,f,f,f,f,f,f,f,f], FenicsMesh.mesh)

        self.width_file = File("width_timeseries.pvd", "compressed")

        #Viscosity Variables
        self.viscosity_type = viscosity_type
        self.viscosity_functions = {
            "glen" : self.calculate_glen_viscosity,
            "glen_calving" : self.calculate_glen_calving_viscosity,
            "newton" : self.calculate_newton_viscosity
        }

        self.creep_parameter = creep_parameter
        self.creep_parameter_exponentiated = creep_parameter**(-1/3)
        self.newton_viscosity = newton_viscosity

        self.epsilon = epsilon
        self.calved_threshold = calved_threshold

    def compute_viscosity(self, velocity, scalar_space, tensor_space, degree):
        return self.viscosity_functions[self.viscosity_type](velocity, scalar_space, tensor_space, degree)

    def calculate_glen_viscosity(self, velocity, scalar_space, tensor_space, degree):
        if not self.creep_parameter:
            raise InputError(self.creep_parameter)

        viscosity_coefficient = 0.5*self.creep_parameter_exponentiated
        strainrate = strain_rate(velocity)
        viscosity = viscosity_coefficient/((0.5*strainrate[0,0]**2 + strainrate[0,1]**2 + 0.5*strainrate[1,1]**2 + self.epsilon)**(1/3))
        return viscosity

    def calculate_glen_calving_viscosity(self, velocity, scalar_space, tensor_space, degree):
        particle_calved = self.particles.return_property(mesh, 9)
        previous_viscosities = self.particles.return_property(mesh, 8)

        viscosity_coefficient = 0.5*self.creep_parameter_exponentiated
        strainrate = strain_rate(velocity)
        viscosity = viscosity_coefficient/((0.5*strainrate[0,0]**2 + strainrate[0,1]**2 + 0.5*strainrate[1,1]**2 + self.epsilon)**(1/3))

        viscosity_projected = project(viscosity, scalar_space)
        self.particles.interpolate(viscosity_projected, 8)
        new_viscosities = self.particles.return_property(mesh, 8)

        failed_ice = (particle_calved > self.calved_threshold)
        for j in range(len(new_viscosities)):
            if failed_ice[j]:
                new_viscosities[j] = previous_viscosities[j]
        
        self.particles.modify_property(new_viscosities, 8)

        return self.get_viscosity()


    def calculate_newton_viscosity(self, velocity, scalar_space, tensor_space, degree):
        if not newton_viscosity:
            raise InputError(newton_viscosity)

        viscosity = Expression("viscosity", viscosity=self.newton_viscosity, degree=degree)
        return viscosity

    def find_calving_location_particle(self, length, time):
        current_width_function = self.get_width()

        x, z = np.transpose(self.particles.return_property(self.FenicsMesh.mesh, 0))
        f = self.particles.return_property(self.FenicsMesh.mesh, 1)
        current_width = interpsci.NearestNDInterpolator(np.transpose([x, z]), f)
        cs = plt.tricontour(x, z, f, [self.yield_width], cmap='Wistia')
        path_list = cs.collections[0].get_paths()

        def calculate_width(x, z):
            try:
                output = current_width(x, z)
            except:
                output = 0.0
            return output

        calving_event = False
        calving_location = 1E15
        calving_bottom = 0.0
        calving_top = 0.0

        [top_boundary_x_full, top_boundary_z_full], [bottom_boundary_x_full, bottom_boundary_z_full] = self.FenicsMesh.get_remeshing_boundaries()
        top_boundary_x = []
        top_boundary_z = []
        bottom_boundary_x = []
        bottom_boundary_z = []
        for j in range(len(top_boundary_x_full)):
            if j == 0:
                top_boundary_x.append(top_boundary_x_full[j])
                top_boundary_z.append(top_boundary_z_full[j])
            else:
                if top_boundary_x_full[0] < top_boundary_x_full[-1]:
                    if top_boundary_x_full[j] > top_boundary_x[-1]:
                        top_boundary_x.append(top_boundary_x_full[j])
                        top_boundary_z.append(top_boundary_z_full[j])
                else:
                    if top_boundary_x_full[j] < top_boundary_x[-1]:
                        top_boundary_x.append(top_boundary_x_full[j])
                        top_boundary_z.append(top_boundary_z_full[j])
        for j in range(len(bottom_boundary_x_full)):
            if j == 0:
                bottom_boundary_x.append(bottom_boundary_x_full[j])
                bottom_boundary_z.append(bottom_boundary_z_full[j])
            else:
                if bottom_boundary_x_full[0] < bottom_boundary_x_full[-1]:
                    if bottom_boundary_x_full[j] > bottom_boundary_x[-1]:
                        bottom_boundary_x.append(bottom_boundary_x_full[j])
                        bottom_boundary_z.append(bottom_boundary_z_full[j])
                else:
                    if bottom_boundary_x_full[j] < bottom_boundary_x[-1]:
                        bottom_boundary_x.append(bottom_boundary_x_full[j])
                        bottom_boundary_z.append(bottom_boundary_z_full[j])

        if top_boundary_x[0] > top_boundary_x[-1]:
            top_boundary_x = top_boundary_x[::-1]
            top_boundary_z = top_boundary_z[::-1]
        if bottom_boundary_x[0] > bottom_boundary_x[-1]:
            bottom_boundary_x = bottom_boundary_x[::-1]
            bottom_boundary_z = bottom_boundary_z[::-1]

        top_interpolator = interpsci.interp1d(top_boundary_x, top_boundary_z, fill_value='extrapolate')
        bot_interpolator = interpsci.interp1d(bottom_boundary_x, bottom_boundary_z, fill_value='extrapolate')

        crevasse_x_locations = []
        bottom_crevasse_locations = []
        surface_crevasse_locations = []

        for path in path_list:
            v = path.vertices
            check_locations = np.unique(v[:, 0])
            for j in range(len(check_locations)):
                full_bottom = True
                full_surface = True
                if check_locations[j] > self.boundary_tolerance:
                    crevasse_x = check_locations[j]
                    corresponding_bottom_z = bot_interpolator(check_locations[j])
                    bottom_crevassse = copy.copy(corresponding_bottom_z)
                    corresponding_surface_z = top_interpolator(check_locations[j])
                    surface_crevasse = copy.copy(corresponding_surface_z)
                    diff = corresponding_surface_z-corresponding_bottom_z
                    num_pts = int((diff)//self.spacing_calving_check)
                    if diff > 0:
                        if calculate_width(check_locations[j], corresponding_bottom_z) >= self.yield_width:
                            if num_pts > 2:
                                calving_check_array_z_lower = np.linspace(corresponding_bottom_z, corresponding_surface_z, num=num_pts)
                            else:
                                calving_check_array_z_lower = np.asarray([corresponding_bottom_z, corresponding_surface_z])
                            for k in range(len(calving_check_array_z_lower)):
                                if calculate_width(check_locations[j], calving_check_array_z_lower[k]) < self.yield_width:
                                    full_bottom = False
                                    break
                            if full_bottom:
                                calving_event = True
                                if check_locations[j] < calving_location:
                                    calving_location = check_locations[j]
                                break

        if calving_event:
            calving_bottom = bot_interpolator(calving_location).item()
            calving_top = top_interpolator(calving_location).item()

        x, z = np.transpose(self.particles.return_property(self.FenicsMesh.mesh, 0))
        f_plot = self.particles.return_property(self.FenicsMesh.mesh, 1)
        sc = plt.scatter(x, z, c=f_plot, s=0.5, cmap='seismic')
        if calving_event:
            plt.scatter(calving_location, calving_bottom, s=10.0, c='k')
            plt.scatter(calving_location, calving_top, s=10.0, c='k')
        plt.colorbar(sc)
        plt.xlim(length-200, length)
        plt.savefig('contours-' + str(time) + '.png')
        if calving_event:
            plt.xlim(calving_location-100, calving_location+100)
            plt.ylim(calving_bottom-10, calving_top+10)
            plt.savefig('contours-calved-' + str(time) + '.png')
        plt.close()

        return [calving_event, [calving_location, calving_bottom, calving_top]], current_width_function

    def manage_tracers(self, calving_info, remesh_check, previous_width):
        if remesh_check:
            x = self.particles.return_property(self.FenicsMesh.mesh, 0)
            f = self.particles.return_property(self.FenicsMesh.mesh, 1)
            if calving_info[0]:
                [xb, zb] = self.FenicsMesh.get_xz_boundary_points()
                polygon = np.transpose([xb, zb])
                path = mpltPath.Path(polygon)
                inside = path.contains_points(x)
                x = x[inside]
                f = f[inside]
            new_particles = particles(x, [f,f,f,f,f,f,f, f, f], self.FenicsMesh.mesh)
            del self.particles
            self.particles = new_particles

        self.add_delete_tracers(previous_width, previous_width, previous_width)

        if not self.advect:
            self.zero_tracers()

    def update_particle_tracers(self, pressure_solution, velocity_solution, dt, length, time):
        # Strain rate
        strainrate = project(strain_rate(velocity_solution), self.FenicsMesh.tensor_space_dg)
        strainrate00 = Function(self.FenicsMesh.scalar_space_dg)
        strainrate01 = Function(self.FenicsMesh.scalar_space_dg)
        strainrate11 = Function(self.FenicsMesh.scalar_space_dg)
        assign(strainrate00, strainrate.sub(0))
        assign(strainrate01, strainrate.sub(1))
        assign(strainrate11, strainrate.sub(3))

        # Full stress
        sigma = project(2*self.compute_viscosity(velocity_solution, self.FenicsMesh.scalar_space_dg, self.FenicsMesh.tensor_space_dg, self.FenicsMesh.degree)*strain_rate(velocity_solution)-pressure_solution*\
                Identity(self.FenicsMesh.tensor_space.ufl_cell().topological_dimension()), self.FenicsMesh.tensor_space_dg)
        sigma00 = Function(self.FenicsMesh.scalar_space_dg)
        sigma01 = Function(self.FenicsMesh.scalar_space_dg)
        sigma11 = Function(self.FenicsMesh.scalar_space_dg)
        assign(sigma00, sigma.sub(0))
        assign(sigma01, sigma.sub(1))
        assign(sigma11, sigma.sub(3))

        x, z = np.transpose(self.particles.return_property(self.FenicsMesh.mesh, 0))
        f = self.particles.return_property(self.FenicsMesh.mesh, 1)

        self.particles.interpolate(strainrate00, 2)
        self.particles.interpolate(strainrate01, 3)
        self.particles.interpolate(strainrate11, 4)
        self.particles.interpolate(sigma00, 5)
        self.particles.interpolate(sigma01, 6)
        self.particles.interpolate(sigma11, 7)

        e00 = self.particles.return_property(self.FenicsMesh.mesh, 2)
        e01 = self.particles.return_property(self.FenicsMesh.mesh, 3)
        e11 = self.particles.return_property(self.FenicsMesh.mesh, 4)
        s00 = self.particles.return_property(self.FenicsMesh.mesh, 5)
        s01 = self.particles.return_property(self.FenicsMesh.mesh, 6)
        s11 = self.particles.return_property(self.FenicsMesh.mesh, 7)

        ddelta = np.sqrt(np.add(np.power((np.subtract(s00,s11)), 2), 4*np.power(s01,2)))
        sigma1 = 0.5*(np.add(np.add(s00,s11),ddelta))
        nye_stress_particles = np.add(sigma1,np.multiply((z<=0.0),(self.rho_w*self.g*(0.0-z))))

        yield_strength_exceeded = (nye_stress_particles > self.yield_strength)
        f = np.maximum(f, yield_strength_exceeded*self.opening_width)

        self.particles.modify_property(f, 1)

    def advect_particles(self, velocity_solution, dt):
        ap = advect_particles(self.particles, self.FenicsMesh.vector_space_quadratic, velocity_solution, "open")
        ap.do_step(dt)

    def get_width(self):
        current_width = Function(self.FenicsMesh.scalar_space_dg)
        width = l2projection(self.particles, self.FenicsMesh.scalar_space_dg, 1)
        width.project(current_width, 0.0, self.opening_width)
        return current_width

    def get_calved(self):
        current_calved = Function(self.FenicsMesh.scalar_space_dg)
        calved = l2projection(self.particles, self.FenicsMesh.scalar_space_dg, 9)
        calved.project(current_calved, 0.0, 1.0)
        return current_calved

    def get_viscosity(self):
        current_viscosity = Function(self.FenicsMesh.scalar_space_dg)
        viscosity = l2projection(self.particles, self.FenicsMesh.scalar_space_dg, 8)

        viscosity_coefficient = 0.5*self.creep_parameter_exponentiated
        upper_bound = viscosity_coefficient/((self.epsilon)**(1/3))

        lower_bound = 2.822E-11 #Viscosity of water in Pa*a

        viscosity.project(current_viscosity, lower_bound, upper_bound)
        return current_viscosity

    def add_delete_tracers(self, width, viscosity, calved):
        AD = AddDelete(self.particles, self.tracers_per_cell, self.tracers_per_cell, [width, width, width, width, width, width, width, viscosity, calved])
        AD.do_sweep()

    def add_delete_tracers_extra(self):
        width = self.get_width()
        viscosity = self.get_width()
        calved = self.get_width()
        AD = AddDelete(self.particles, self.tracers_per_cell, self.tracers_per_cell, [width, width, width, width, width, width, width, viscosity, calved])
        AD.do_sweep()

    def zero_tracers(self):
        zero = interpolate(Constant(0.0), self.FenicsMesh.scalar_space_dg)
        self.particles.interpolate(zero, 1)

    def save_width(self, time):
        current_width = self.get_width()
        current_width.rename('width', 'width')

        self.width_file << (current_width, time)
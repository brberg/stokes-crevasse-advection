from __future__ import division
import time as timelib
import numpy as np
import copy
from dolfin import  *
from viscosity import *
from tracers import *
from ufl import nabla_div

#Only display fenics warnings. Supress simple information.
set_log_level(20)
parameters["allow_extrapolation"] = True

def print_time(start_time, message=''):
    end_time = timelib.time()
    elapsed_time = int(round(end_time - start_time))
    print(message + ':' + str(elapsed_time))

class SimulationSolver:

    seconds_in_year = Constant(3.154E7)

    def __init__(self, FenicsMesh, friction_coefficient, solver_tolerance, sea_spring=False, shear_calving=False, shear_angle=45, shear_strength=165000, calving=True, advect=True, calving_type = 'particle', yield_strength=0.0, yield_width = 0.0099, opening_width=0.01, smoothing_width=1000, calving_start = 0.0, tracers_per_cell=32, inflow = False, inflow_velocity = 0.0, inflow_bottom = None, inflow_top = None, viscosity_type = 'glen', creep_parameter = False, newton_viscosity=False, remesh_always=False, save_all=False, save_mesh=True, smb_type='fenics', solver_type = 'linear_problem', linear_solver = 'mumps', preconditioner = 'None', rho_i=900, rho_w=1000, rho_s = 1E5, g=9.81, epsilon=DOLFIN_EPS, melt_type='constant', avg_melt=0.0, coulomb=False):

        #Sub-Classes
        self.FenicsMesh = FenicsMesh
        self.viscosity_type = viscosity_type
        self.creep_parameter = creep_parameter
        self.newton_viscosity = newton_viscosity

        #Sea-spring
        self.sea_spring = sea_spring

        #Calving 
        if self.FenicsMesh.steady_state:
            self.calving = False
        else:
            self.calving = calving

        if coulomb:
            chosen_yield_strength = FenicsMesh.KIc/(np.pi*FenicsMesh.c)**(1/2)
        else:
            chosen_yield_strength = yield_strength
        self.tensile_strength = chosen_yield_strength
        self.Tracers = Tracers(FenicsMesh, rho_i, rho_w, g, advect, calving_type, chosen_yield_strength, yield_width, opening_width, smoothing_width, tracers_per_cell, FenicsMesh.hires_cell_size/tracers_per_cell, FenicsMesh.boundary_tolerance, viscosity_type, creep_parameter, newton_viscosity, epsilon)
        self.calving_start = calving_start

        #Idealized Shear Calving
        self.shear_calving = shear_calving
        self.shear_angle = shear_angle
        self.shear_strength = shear_strength
        self.single_shear = True

        #Physical inputs
        self.friction_coefficient = friction_coefficient

        #Model options
        self.inflow = inflow
        self.inflow_top = inflow_top
        self.inflow_bottom = inflow_bottom
        self.remesh_always = remesh_always
        self.save_all = save_all
        self.save_mesh = save_mesh

        #Type of solver implmentation for Fenics
        self.solver_type = solver_type
        self.solver_functions = {
            'solve' : self.fenics_solution_solve,
            'linear_problem' : self.fenics_solution_linear_problem,
        }
        self.smb_type = smb_type
        self.smb_functions = {
            'fenics' : self.solve_smb,
            'numpy' : self.solve_smb_numpy,
            'fenics_new' : self.solve_smb_new,
            'numpy_new' : self.solve_smb_numpy_new
        }
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner
        #Numerical Constants
        self.solver_tolerance = solver_tolerance
        self.epsilon = epsilon

        #Define variable to track simulation time 
        self.time = 0.0

        #Physical Constants
        self.rho_i = rho_i
        self.rho_w = rho_w
        self.rho_s = rho_s
        self.g = g

        #Velocities and pressure used in solver loops
        self.iteration_velocity = interpolate(Constant((0.0, 0.0)), FenicsMesh.vector_space_quadratic)
        self.step_velocity = interpolate(Constant((FenicsMesh.inflow_velocity, 0.0)), FenicsMesh.vector_space_quadratic)

        #Total solution to track between time steps and remeshing
        self.solver_solution = Function(FenicsMesh.mixed_space)

        #Physical Fenics Expressions
        self.body_force = Constant((0, -rho_i*g))

        class WaterPressure(UserExpression):
            def eval(self, value, x):
                if x[1] <= 0.0:
                    value[0] = rho_w*g*(0.0-x[1])
                else:
                    value[0] = 0.0
        self.water_pressure_expression = WaterPressure(degree=FenicsMesh.degree)

        class GroundPressure(UserExpression):
            def eval(self, value, x):
                if x[1] <= FenicsMesh.bed_interpolator(x[0]):
                    value[0] = rho_s*g*(FenicsMesh.bed_interpolator(x[0])-x[1])
                else:
                    value[0] = 0.0
        self.ground_pressure_expression = GroundPressure(degree=FenicsMesh.degree)

        class SurfaceMassBalance(UserExpression):
            def eval(self, value, x):
                value[0] = 0.0
                if x[1] <= 0.0 or x[1] <= (FenicsMesh.bed_interpolator(x[0]) + FenicsMesh.boundary_tolerance):
                    value[1] = 0.0
                else:
                    value[1] = FenicsMesh.smb_interpolator(x[0])
            def value_shape(self):
                return (2,)
        self.smb_expression = SurfaceMassBalance(degree=FenicsMesh.degree)

        class SurfaceMassBalanceZ(UserExpression):
            def eval(self, value, x):
                if x[1] <= 0.0 or x[1] <= (FenicsMesh.bed_interpolator(x[0]) + FenicsMesh.boundary_tolerance):
                    value[0] = 0.0
                else:
                    value[0] = FenicsMesh.smb_interpolator(x[0])
        self.smb_expression_z = SurfaceMassBalanceZ(degree=FenicsMesh.degree)


        D = FenicsMesh.bed_interpolator(50000)
        avg_melt = avg_melt*365.25
        self.avg_melt = avg_melt
        gl_position = FenicsMesh.grounding_line_location()
        class ConstantMelt(UserExpression):
            def eval(self, value, x):
                if x[1] > 0.0 or x[0] < gl_position:
                    value[0] = 0.0
                else:
                    value[0] = avg_melt
        class LinearMelt(UserExpression):
            def eval(self, value, x):
                if x[1] > 0.0 or x[0] < gl_position:
                    value[0] = 0.0
                else:
                    value[0] = 2*avg_melt*(x[1]/D)
        class ParabolicMelt(UserExpression):
            def eval(self, value, x):
                if x[1] > 0.0 or x[0] < gl_position:
                    value[0] = 0.0
                else:
                    value[0] = 6*avg_melt*(x[1]/D)*(1-(x[1]/D))

        self.melt_type = melt_type
        if melt_type == 'constant':
            self.melt_expression = ConstantMelt(degree=FenicsMesh.degree)
        elif melt_type == 'linear':
            self.melt_expression = LinearMelt(degree=FenicsMesh.degree)
        elif melt_type == 'parabolic':
            self.melt_expression = ParabolicMelt(degree=FenicsMesh.degree)
        else:
            raise InputError(melt_type)
        File("melt_profile.pvd") << interpolate(self.melt_expression, FenicsMesh.scalar_space_linear)

        #Setup files for saving velocity and pressure solutions
        self.ux_file = File("ux_timeseries.pvd", "compressed")
        self.uz_file = File("uz_timeseries.pvd", "compressed")
        self.p_file = File("p_timeseries.pvd", "compressed")

        #Setup files for saving derived physical quantities
        self.udiv_file = File("udiv_timeseries.pvd", "compressed")
        self.viscosity_file = File("viscosity_timeseries.pvd", "compressed")
        self.strainrate00_file = File("strainrate00_timeseries.pvd", "compressed")
        self.strainrate01_file = File("strainrate01_timeseries.pvd", "compressed")
        self.strainrate11_file = File("strainrate11_timeseries.pvd", "compressed")
        self.sigma00_file = File("sigma00_timeseries.pvd", "compressed")
        self.sigma01_file = File("sigma01_timeseries.pvd", "compressed")
        self.sigma11_file = File("sigma11_timeseries.pvd", "compressed")
        self.tau00_file = File("tau00_timeseries.pvd", "compressed")
        self.tau01_file = File("tau01_timeseries.pvd", "compressed")
        self.tau11_file = File("tau11_timeseries.pvd", "compressed")
        self.nye_file = File("nyestress_timeseries.pvd", "compressed")
        self.strainrateE_file = File("strainrateE_timeseries.pvd", "compressed")
        self.sigmaP_file = File("sigmaP_timeseries.pvd", "compressed")
        self.tauMax_file = File("tauMax_timeseries.pvd", "compressed")

        #Setup files for saving normal vectors and accumulation
        self.normalvec_file = File("normal_vec.pvd", "compressed")
        self.normalvec2_file = File("normal_vec_averaged.pvd", "compressed")
        self.normalx_file = File("normal_x.pvd", "compressed")
        self.normalz_file = File("normal_z.pvd", "compressed")
        self.accumulation_file = File("accumulation.pvd", "compressed")
        self.meltx_file = File("melt_x.pvd", "compressed")
        self.meltz_file = File("melt_z.pvd", "compressed")

    def redefine_melt_expressions(self):
        [x, z] = self.FenicsMesh.get_xz_boundary_points()
        max_length = max(x)
        D = self.FenicsMesh.bed_interpolator(max_length)
        avg_melt = self.avg_melt
        gl_position = self.FenicsMesh.grounding_line_location()
        FenicsMesh = self.FenicsMesh
        class ConstantMelt(UserExpression):
            def eval(self, value, x):
                if x[1] > 0.0 or x[0] < gl_position:
                    value[0] = 0.0
                else:
                    value[0] = avg_melt
        class LinearMelt(UserExpression):
            def eval(self, value, x):
                if x[1] > 0.0 or x[0] < gl_position:
                    value[0] = 0.0
                else:
                    value[0] = 2*avg_melt*(x[1]/D)
        class ParabolicMelt(UserExpression):
            def eval(self, value, x):
                if x[1] > 0.0 or x[0] < gl_position:
                    value[0] = 0.0
                else:
                    value[0] = 6*avg_melt*(x[1]/D)*(1-(x[1]/D))

        if self.melt_type == 'constant':
            self.melt_expression = ConstantMelt(degree=self.FenicsMesh.degree)
        elif self.melt_type == 'linear':
            self.melt_expression = LinearMelt(degree=self.FenicsMesh.degree)
        elif self.melt_type == 'parabolic':
            self.melt_expression = ParabolicMelt(degree=self.FenicsMesh.degree)
        else:
            raise InputError(self.melt_type)

    def b_function(self, v, q):
        return inner(nabla_div(v), q)*dx

    def norm_calculator(self, velocity):
        return assemble(inner(velocity, velocity)*dx)

    def save_xml_mesh(self):
        File('mesh_' + str(self.time) + '.xml') << self.FenicsMesh.mesh

    def get_shear_stresses(self):
        pressure_solution, velocity_solution = self.solver_solution.split()

        tau_raw = 2*self.Tracers.compute_viscosity(velocity_solution, self.FenicsMesh.scalar_space_dg, self.FenicsMesh.tensor_space_dg, self.FenicsMesh.degree)*strain_rate(velocity_solution)
        tau_max_ufl = (0.5*tau_raw[0,0]**2+0.5*tau_raw[1,1]**2+tau_raw[0,1]**2)**(1/2)
        tau_max = project(tau_max_ufl, self.FenicsMesh.scalar_space_dg)

        tau = project(tau_raw, self.FenicsMesh.tensor_space_dg)
        tau00 = Function(self.FenicsMesh.scalar_space_dg)
        tau01 = Function(self.FenicsMesh.scalar_space_dg)
        tau11 = Function(self.FenicsMesh.scalar_space_dg)
        assign(tau00, tau.sub(0))
        assign(tau01, tau.sub(1))
        assign(tau11, tau.sub(3))

        return tau_max, tau00, tau01, tau11

    def get_coulomb_stresses(self):
        pressure_solution, velocity_solution = self.solver_solution.split()

        sigma_raw = 2*self.Tracers.compute_viscosity(velocity_solution, self.FenicsMesh.scalar_space_dg, self.FenicsMesh.tensor_space_dg, self.FenicsMesh.degree)*strain_rate(velocity_solution)-pressure_solution*\
                Identity(self.FenicsMesh.tensor_space.ufl_cell().topological_dimension())

        sigma = project(sigma_raw, self.FenicsMesh.tensor_space_dg)
        sigma00 = Function(self.FenicsMesh.scalar_space_dg)
        sigma01 = Function(self.FenicsMesh.scalar_space_dg)
        sigma11 = Function(self.FenicsMesh.scalar_space_dg)
        assign(sigma00, sigma.sub(0))
        assign(sigma01, sigma.sub(1))
        assign(sigma11, sigma.sub(3))

        return sigma00, sigma01, sigma11


    def save_solution(self):
        pressure_solution, velocity_solution = self.solver_solution.split()
        velocity_x_solution, velocity_z_solution = velocity_solution.split()

        velocity_x_solution.rename('ux', 'ux')
        velocity_z_solution.rename('uz', 'uz')
        pressure_solution.rename('p', 'p')

        self.ux_file << (velocity_x_solution, self.time)
        self.uz_file << (velocity_z_solution, self.time)
        self.p_file << (pressure_solution, self.time)

    def save_solution_extra(self):
        pressure_solution, velocity_solution = self.solver_solution.split()

        udiv = project(nabla_div(velocity_solution), self.FenicsMesh.scalar_space_dg)

        viscosity = project(self.Tracers.compute_viscosity(velocity_solution, self.FenicsMesh.scalar_space_dg, self.FenicsMesh.tensor_space_dg, self.FenicsMesh.degree), self.FenicsMesh.scalar_space_dg)

        strainrate = project(strain_rate(velocity_solution), self.FenicsMesh.tensor_space_dg)
        strainrate00 = Function(self.FenicsMesh.scalar_space_dg)
        strainrate01 = Function(self.FenicsMesh.scalar_space_dg)
        strainrate11 = Function(self.FenicsMesh.scalar_space_dg)
        assign(strainrate00, strainrate.sub(0))
        assign(strainrate01, strainrate.sub(1))
        assign(strainrate11, strainrate.sub(3))

        # Full stress
        sigma = project(2*self.Tracers.compute_viscosity(velocity_solution, self.FenicsMesh.scalar_space_dg, self.FenicsMesh.tensor_space_dg, self.FenicsMesh.degree)*strain_rate(velocity_solution)-pressure_solution*\
                Identity(self.FenicsMesh.tensor_space.ufl_cell().topological_dimension()), self.FenicsMesh.tensor_space_dg)
        sigma00 = Function(self.FenicsMesh.scalar_space_dg)
        sigma01 = Function(self.FenicsMesh.scalar_space_dg)
        sigma11 = Function(self.FenicsMesh.scalar_space_dg)
        assign(sigma00, sigma.sub(0))
        assign(sigma01, sigma.sub(1))
        assign(sigma11, sigma.sub(3))

        tau_raw = 2*self.Tracers.compute_viscosity(velocity_solution, self.FenicsMesh.scalar_space_dg, self.FenicsMesh.tensor_space_dg, self.FenicsMesh.degree)*strain_rate(velocity_solution)
        tau_max_ufl = (0.5*tau_raw[0,0]**2+0.5*tau_raw[1,1]**2+tau_raw[0,1]**2)**(1/2)
        tau_max = project(tau_max_ufl, self.FenicsMesh.scalar_space_dg)

        tau = project(tau_raw, self.FenicsMesh.tensor_space_dg)
        tau00 = Function(self.FenicsMesh.scalar_space_dg)
        tau01 = Function(self.FenicsMesh.scalar_space_dg)
        tau11 = Function(self.FenicsMesh.scalar_space_dg)
        assign(tau00, tau.sub(0))
        assign(tau01, tau.sub(1))
        assign(tau11, tau.sub(3))

        ddelta = Expression("sqrt(pow((sigmaxx-sigmazz), 2) + 4*pow(sigmaxz, 2))",
                                sigmaxx=sigma00, sigmaxz=sigma01, sigmazz=sigma11, degree=self.FenicsMesh.degree)
        sigma1 = Expression("0.5*(sigmaxx + sigmazz + ddelta)", sigmaxx=sigma00,
                                        sigmazz=sigma11, ddelta=ddelta,
                                        degree=self.FenicsMesh.degree)
        sigmaP = project(sigma1, self.FenicsMesh.scalar_space_dg)
        interpolated_nye_stress = project(Expression(("(sigma1+(x[1]<=0)*(A*B*(0 - x[1])))"), sigma1=sigma1, A=self.rho_w, B=self.g, degree=self.FenicsMesh.degree), self.FenicsMesh.scalar_space_dg)

        epsilonE = project(Expression(("sqrt(0.5*pow(e00,2)+0.5*pow(e11,2)+pow(e01,2))"), e00=strainrate00, e01=strainrate01, e11=strainrate11, degree=self.FenicsMesh.degree), self.FenicsMesh.scalar_space_dg)

        udiv.rename('udiv', 'udiv')
        viscosity.rename('viscosity', 'viscosity')
        strainrate00.rename('e00', 'e00')
        strainrate01.rename('e01', 'e01')
        strainrate11.rename('e11', 'e11')
        sigma00.rename('s00', 's00')
        sigma01.rename('s01', 's01')
        sigma11.rename('s11', 's11')
        tau00.rename('t00', 't00')
        tau01.rename('t01', 't01')
        tau11.rename('t11', 't11')
        interpolated_nye_stress.rename('s', 's')
        epsilonE.rename('e', 'e')
        sigmaP.rename('sP', 'sP')
        tau_max.rename('tM', 'tM')

        self.udiv_file << (udiv, self.time)
        self.viscosity_file << (viscosity, self.time)
        self.strainrate00_file << (strainrate00, self.time)
        self.strainrate01_file << (strainrate01, self.time)
        self.strainrate11_file << (strainrate11, self.time)
        self.sigma00_file << (sigma00, self.time)
        self.sigma01_file << (sigma01, self.time)
        self.sigma11_file << (sigma11, self.time)
        self.tau00_file << (tau00, self.time)
        self.tau01_file << (tau01, self.time)
        self.tau11_file << (tau11, self.time)
        self.nye_file << (interpolated_nye_stress, self.time)
        self.strainrateE_file << (epsilonE, self.time)
        self.sigmaP_file << (sigmaP, self.time)
        self.tauMax_file << (tau_max, self.time)


    def fenics_solution(self, variational_form):
        self.solver_functions[self.solver_type](variational_form)

    def fenics_solution_solve(self, variational_form):
        solve(lhs(variational_form) == rhs(variational_form), self.solver_solution, self.FenicsMesh.boundary_conditions)

    def fenics_solution_linear_problem(self, variational_form):
        problem = LinearVariationalProblem(lhs(variational_form), rhs(variational_form), self.solver_solution, self.FenicsMesh.boundary_conditions)
        solver = LinearVariationalSolver(problem)
        solver.parameters['linear_solver'] = self.linear_solver
        solver.parameters['preconditioner'] = self.preconditioner
        solver.solve()

    def compute_verification_functions(self):
        class AnalyticVelocityX(UserExpression):
            def eval(self, value, x):
                value[0] = x[0] + np.power(x[0], 2) - 2*x[0]*x[1] + np.power(x[0], 3) - 3*x[0]*np.power(x[1], 2) + np.power(x[0], 2)*x[1]

        class AnalyticVelocityZ(UserExpression):
            def eval(self, value, x):
                value[0] = -x[1] + np.power(x[1], 2) - 2*x[1]*x[0] + np.power(x[1], 3) - 3*x[1]*np.power(x[0], 2) - np.power(x[1], 2)*x[0]

        class AnalyticPressure(UserExpression):
            def eval(self, value, x):
                value[0] = x[0]*x[1] + x[0] + x[1] + np.power(x[0], 3)*np.power(x[1], 2) - 4/3

        if self.viscosity_type == 'glen':
            class AnalyticForcing(UserExpression):
                def eval(self, value, x):
                    n = 3

                    g = 2 + 4*x[0]-4*x[1]+6*np.power(x[0], 2)-6*np.power(x[1], 2)+4*x[0]*x[1]
                    h = -2*x[0] - 12*x[0]*x[1] + np.power(x[0], 2) - 2*x[1] - np.power(x[1], 2)
                    dgdx = 4+12*x[0]+4*x[1]
                    dgdz = -4-12*x[1]+4*x[0]
                    dhdx = -2-12*x[1]+2*x[0]
                    dhdz = -12*x[0]-2-2*x[1]


                    prefactor1 = ((1-n)/(4*n))
                    prefactor2 = np.power((np.power(g, 2)+np.power(h, 2))/4, (1-3*n)/(2*n))
                    prefactor3 = np.power((np.power(g, 2)+np.power(h, 2))/4, (1-n)/(2*n))

                    value[0] = -prefactor1*prefactor2*(g*dgdx+h*dhdx)*g \
                    -prefactor1*prefactor2*(g*dgdz+h*dhdz)*h \
                    -(2+2*x[1])*prefactor3+1+x[1]+3*np.power(x[0], 2)*np.power(x[1], 2)

                    value[1] = prefactor1*prefactor2*(g*dgdz+h*dhdz)*g \
                    -prefactor1*prefactor2*(g*dgdx+h*dhdx)*h \
                    -(2-2*x[0])*prefactor3+1+x[0]+2*np.power(x[0], 3)*x[1]
                def value_shape(self):
                    return (2,)
        elif self.viscosity_type == 'newton':
            newton_viscosity = self.newton_viscosity
            rho_i = self.rho_i
            class AnalyticForcing(UserExpression):
                def eval(self, value, x):
                    value[0] = -newton_viscosity*(2+6*x[0]+2*x[1]-6*x[0]) + x[1] + 1 + 3*np.power(x[0], 2)*np.power(x[1], 2)
                    value[1] = -newton_viscosity*(-6*x[1]+2+6*x[1]-2*x[0]) + x[0] + 1 + 2*np.power(x[0], 3)*np.power(x[1], 1)
                def value_shape(self):
                    return (2,)

        self.analytic_velocity_x = AnalyticVelocityX(degree=self.FenicsMesh.degree+1)
        self.analytic_velocity_z = AnalyticVelocityZ(degree=self.FenicsMesh.degree+1)
        self.analytic_pressure = AnalyticPressure(degree=self.FenicsMesh.degree)
        self.analytic_forcing = AnalyticForcing(degree=self.FenicsMesh.degree)

        self.analytic_velocity_x = AnalyticVelocityX(degree=self.FenicsMesh.degree+1)
        File("verification_analytic_ux.pvd") << project(self.analytic_velocity_x, self.FenicsMesh.scalar_space_quadratic)
        File("verification_analytic_uz.pvd") << project(self.analytic_velocity_z,self.FenicsMesh.scalar_space_quadratic)
        File("verification_analytic_p.pvd") << project(self.analytic_pressure, self.FenicsMesh.scalar_space_linear)

    def run_model(self, duration, dt):

        while self.time < duration:
            start_time = timelib.time()

            if self.sea_spring:
                dt_cfl = self.solve_model_ss(dt)
            else:
                dt_cfl = self.solve_model(dt)
            pressure_solution, velocity_solution = self.solver_solution.split()
            velocity_x_solution, velocity_z_solution = velocity_solution.split()

            self.save_solution()
            if self.save_all:
                self.save_solution_extra()
            if self.save_mesh:
                self.save_xml_mesh()

            boundary_points = False
            calving_x = 0.0
            shear_calving_event = False
            if self.shear_calving and self.time >= self.calving_start:
                sigma00, sigma01, sigma11 = self.get_coulomb_stresses()
                boundary_points, calving_x, shear_calving_event = self.FenicsMesh.calculate_shear_calving_event(sigma00, sigma01, sigma11, self.tensile_strength, self.Tracers)
                if shear_calving_event:
                    self.single_shear = False
            if self.calving and self.time >= self.calving_start:
                self.Tracers.update_functions[self.Tracers.calving_type](pressure_solution, velocity_solution, dt_cfl, self.FenicsMesh.mesh.coordinates()[:, 0].max(), self.time)

            ux = velocity_x_solution.compute_vertex_values()
            uz = velocity_z_solution.compute_vertex_values()
            if self.calving or self.shear_calving:
                temp_width = self.Tracers.get_width()
            if self.smb_type == 'fenics_new' or self.smb_type == 'numpy_new':
                File("b1.pvd") << self.FenicsMesh.mesh
                self.FenicsMesh.mesh.coordinates()[:, 0] = self.FenicsMesh.mesh.coordinates()[:, 0] + ux*dt_cfl
                self.FenicsMesh.mesh.coordinates()[:, 1] = self.FenicsMesh.mesh.coordinates()[:, 1] + uz*dt_cfl
                if self.calving or self.shear_calving:
                    self.Tracers.advect_particles(velocity_solution, dt_cfl)
                File("b2.pvd") << self.FenicsMesh.mesh
                new_bmesh = self.smb_functions[self.smb_type](dt_cfl)
                ALE.move(self.FenicsMesh.mesh, new_bmesh)
                File("b3.pvd") << self.FenicsMesh.mesh
            else:
                uz_smb = self.smb_functions[self.smb_type]()
                self.FenicsMesh.mesh.coordinates()[:, 0] = self.FenicsMesh.mesh.coordinates()[:, 0] + ux*dt_cfl
                self.FenicsMesh.mesh.coordinates()[:, 1] = self.FenicsMesh.mesh.coordinates()[:, 1] + uz*dt_cfl + uz_smb*dt_cfl
                if self.calving or self.shear_calving:
                    self.Tracers.advect_particles(velocity_solution, dt_cfl)
            if self.calving or self.shear_calving:
                self.Tracers.particles.relocate()
                self.Tracers.add_delete_tracers(temp_width, temp_width, temp_width)
        

            calving_info = [False, [1E15, 0.0, 0.0]]

            self.FenicsMesh.mesh.bounding_box_tree().build(self.FenicsMesh.mesh)
            if self.calving:
                if self.time >= self.calving_start:
                    calving_info, previous_width = self.Tracers.calving_functions[self.Tracers.calving_type](self.FenicsMesh.mesh.coordinates()[:, 0].max(), self.time)
                else:
                    previous_width = self.Tracers.get_width()
                self.Tracers.save_width(self.time)
            if self.FenicsMesh.steady_state:
                calving_info = self.FenicsMesh.set_steady_state_calving()

            self.FenicsMesh.mesh.bounding_box_tree().build(self.FenicsMesh.mesh)
            remesh_check = self.remesh_always or self.inflow or calving_info[0] or MeshQuality.radius_ratio_min_max(self.FenicsMesh.mesh)[0] < 0.1

            if shear_calving_event:
                self.FenicsMesh.shear_remesh(boundary_points)
                self.iteration_velocity = interpolate(Constant((0.0, 0.0)), self.FenicsMesh.vector_space_quadratic)
                self.step_velocity = interpolate(Constant((0.0, 0.0)), self.FenicsMesh.vector_space_quadratic)

                self.FenicsMesh.mesh.bounding_box_tree().build(self.FenicsMesh.mesh)
                self.solver_solution = interpolate(self.solver_solution, self.FenicsMesh.mixed_space)
                pressure_solution_interpolated, velocity_solution_interpolated = self.solver_solution.split()
                assign(self.iteration_velocity, velocity_solution_interpolated)
                assign(self.step_velocity, velocity_solution_interpolated)
            elif remesh_check:
                if self.inflow:
                    self.FenicsMesh.remesh_inflow(self.inflow_bottom, self.inflow_top, calving_info)
                elif calving_info[0]:
                    self.FenicsMesh.remesh_old(calving_info)
                else:
                    self.FenicsMesh.remesh(calving_info)

                self.iteration_velocity = interpolate(Constant((0.0, 0.0)), self.FenicsMesh.vector_space_quadratic)
                self.step_velocity = interpolate(Constant((0.0, 0.0)), self.FenicsMesh.vector_space_quadratic)

                self.FenicsMesh.mesh.bounding_box_tree().build(self.FenicsMesh.mesh)
                self.solver_solution = interpolate(self.solver_solution, self.FenicsMesh.mixed_space)
                pressure_solution_interpolated, velocity_solution_interpolated = self.solver_solution.split()
                assign(self.iteration_velocity, velocity_solution_interpolated)
                assign(self.step_velocity, velocity_solution_interpolated)
            else:
                self.FenicsMesh.define_boundaries()
                assign(self.step_velocity, velocity_solution)

            if shear_calving_event:
                fake_calving_info = [True, [calving_x, calving_x, calving_x]]
                self.Tracers.manage_tracers(fake_calving_info, True, previous_width)
            elif self.calving:
                self.Tracers.manage_tracers(calving_info, remesh_check, previous_width)
            self.time += dt_cfl
            self.redefine_melt_expressions()

            end_time = timelib.time()
            elapsed_time = int(round(end_time - start_time))

            print("Code has finished time step and took " + str(elapsed_time) + " seconds. Current time in years is " + str(self.time) + "/" + str(duration) + ".")

    def solve_model(self, dt):

        dt_cfl = dt
        dt_old = DOLFIN_EPS

        trial_functions = TrialFunction(self.FenicsMesh.mixed_space)
        test_functions = TestFunction(self.FenicsMesh.mixed_space)
        (p, u) = split(trial_functions)
        (q, v) = split(test_functions)

        epsilon_vector = project(Constant((self.epsilon, self.epsilon)), self.FenicsMesh.vector_space_quadratic)
        z_hat = interpolate(Constant((0.0, 1.0)), self.FenicsMesh.vector_space_quadratic)
        facet_normal = FacetNormal(self.FenicsMesh.mesh)
        n_hat = as_vector([facet_normal[0], facet_normal[1]])
        t_hat = as_vector([-facet_normal[1], facet_normal[0]])


        while abs((dt_cfl-dt_old)/(dt_old)) > 0.01:
            norm = 1.0
            while float(norm) >= self.solver_tolerance:
                iteration_viscosity = self.Tracers.compute_viscosity(self.iteration_velocity, self.FenicsMesh.scalar_space_dg, self.FenicsMesh.tensor_space_dg, self.FenicsMesh.degree)
                
                variational_form = 2*inner(iteration_viscosity*strain_rate(u), strain_rate(v))*dx - self.b_function(v, p) + self.b_function(u, q) \
                    + inner(self.rho_i*1/(self.seconds_in_year*dt_cfl)*(u-self.step_velocity), v)*dx \
                    - inner(self.body_force, v)*dx \
                    + inner(self.friction_coefficient*abs(dot(self.iteration_velocity + epsilon_vector, t_hat))**(-2/3)*dot(u, t_hat)*t_hat, v)*self.FenicsMesh.ds(1) \
                    - inner(self.rho_s*self.g*dt_cfl*dot(u, z_hat)*n_hat, v)*self.FenicsMesh.ds(1) \
                    + inner(self.ground_pressure_expression*n_hat, v)*self.FenicsMesh.ds(1) \
                    - inner(self.rho_w*self.g*dt_cfl*dot(u, z_hat)*n_hat, v)*self.FenicsMesh.ds(2) \
                    + inner(self.water_pressure_expression*n_hat, v)*self.FenicsMesh.ds(2) \
                    + inner(self.friction_coefficient*abs(dot(self.iteration_velocity + epsilon_vector, t_hat))**(-2/3)*dot(u, t_hat)*t_hat, v)*self.FenicsMesh.ds(3) \
                    - inner(self.rho_s*self.g*dt_cfl*dot(u, z_hat)*n_hat, v)*self.FenicsMesh.ds(3) \
                    + inner(self.ground_pressure_expression*n_hat, v)*self.FenicsMesh.ds(3) \
                    - inner(self.rho_w*self.g*dt_cfl*dot(u, z_hat)*n_hat, v)*self.FenicsMesh.ds(3) \
                    + inner(self.water_pressure_expression*n_hat, v)*self.FenicsMesh.ds(3)

                self.fenics_solution(variational_form)

                pressure_solution, velocity_solution = self.solver_solution.split()
                velocity_x_solution, velocity_z_solution = velocity_solution.split()

                velocity_solution_change = velocity_solution - self.iteration_velocity

                norm = self.norm_calculator(velocity_solution_change)/self.norm_calculator(velocity_solution)

                #Update velocity for next iteration
                assign(self.iteration_velocity, velocity_solution)

                #Print out progress
                ux_max = np.max(interpolate(velocity_x_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
                ux_min = np.min(interpolate(velocity_x_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
                uz_max = np.max(interpolate(velocity_z_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
                uz_min = np.min(interpolate(velocity_z_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
                p_max = np.max(interpolate(pressure_solution, self.FenicsMesh.scalar_space_linear).vector().get_local())
                p_min = np.min(interpolate(pressure_solution, self.FenicsMesh.scalar_space_linear).vector().get_local())
                # print("X-Velocity Range = [%G, %G]" % (ux_min, ux_max))
                # print("Z-Velocity Range = [%G, %G]" % (uz_min, uz_max))
                # print("Pressure Range = [%G, %G]" % (p_min, p_max))

                # print("Norm is " + str(float(norm)) + ".")

            dt_old = copy.copy(dt_cfl)

            cfl_dt_limit = project(0.25*(CellDiameter(self.FenicsMesh.mesh))/(sqrt(inner(velocity_solution, velocity_solution))+self.avg_melt), self.FenicsMesh.scalar_space_quadratic).compute_vertex_values()

            cfl_dt_limit_no_overflow = []
            for j in range(len(cfl_dt_limit)):
                if cfl_dt_limit[j] > 0.0:
                    cfl_dt_limit_no_overflow.append(cfl_dt_limit[j])

            dt_cfl = min(dt_old, np.amin(cfl_dt_limit_no_overflow))

            print("New time step is " + str(dt_cfl) + " and old time step is " + str(dt_old) + ".")

        return dt_cfl

    def solve_model_ss(self, dt):

        dt_cfl = dt
        dt_old = DOLFIN_EPS

        trial_functions = TrialFunction(self.FenicsMesh.mixed_space)
        test_functions = TestFunction(self.FenicsMesh.mixed_space)
        (p, u) = split(trial_functions)
        (q, v) = split(test_functions)

        epsilon_vector = project(Constant((self.epsilon, self.epsilon)), self.FenicsMesh.vector_space_quadratic)
        z_hat = interpolate(Constant((0.0, 1.0)), self.FenicsMesh.vector_space_quadratic)
        facet_normal = FacetNormal(self.FenicsMesh.mesh)
        n_hat = as_vector([facet_normal[0], facet_normal[1]])
        t_hat = as_vector([-facet_normal[1], facet_normal[0]])


        while abs((dt_cfl-dt_old)/(dt_old)) > 0.01:
            norm = 1.0
            while float(norm) >= self.solver_tolerance:
                iteration_viscosity = self.Tracers.compute_viscosity(self.iteration_velocity, self.FenicsMesh.scalar_space_dg, self.FenicsMesh.tensor_space_dg, self.FenicsMesh.degree)
                
                variational_form = 2*inner(iteration_viscosity*strain_rate(u), strain_rate(v))*dx - self.b_function(v, p) + self.b_function(u, q) \
                    - inner(self.body_force, v)*dx \
                    + inner(self.friction_coefficient*abs(dot(self.iteration_velocity + epsilon_vector, t_hat))**(-2/3)*dot(u, t_hat)*t_hat, v)*self.FenicsMesh.ds(1) \
                    - inner(self.rho_s*self.g*dt_cfl*dot(u, z_hat)*n_hat, v)*self.FenicsMesh.ds(1) \
                    + inner(self.ground_pressure_expression*n_hat, v)*self.FenicsMesh.ds(1) \
                    - inner(self.rho_w*self.g*dt_cfl*dot(u, z_hat)*n_hat, v)*self.FenicsMesh.ds(2) \
                    + inner(self.water_pressure_expression*n_hat, v)*self.FenicsMesh.ds(2) \
                    + inner(self.friction_coefficient*abs(dot(self.iteration_velocity + epsilon_vector, t_hat))**(-2/3)*dot(u, t_hat)*t_hat, v)*self.FenicsMesh.ds(3) \
                    - inner(self.rho_s*self.g*dt_cfl*dot(u, z_hat)*n_hat, v)*self.FenicsMesh.ds(3) \
                    + inner(self.ground_pressure_expression*n_hat, v)*self.FenicsMesh.ds(3) \
                    - inner(self.rho_w*self.g*dt_cfl*dot(u, z_hat)*n_hat, v)*self.FenicsMesh.ds(3) \
                    + inner(self.water_pressure_expression*n_hat, v)*self.FenicsMesh.ds(3)

                self.fenics_solution(variational_form)

                pressure_solution, velocity_solution = self.solver_solution.split()
                velocity_x_solution, velocity_z_solution = velocity_solution.split()

                velocity_solution_change = velocity_solution - self.iteration_velocity

                norm = self.norm_calculator(velocity_solution_change)/self.norm_calculator(velocity_solution)

                #Update velocity for next iteration
                assign(self.iteration_velocity, velocity_solution)

                #Print out progress
                ux_max = np.max(interpolate(velocity_x_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
                ux_min = np.min(interpolate(velocity_x_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
                uz_max = np.max(interpolate(velocity_z_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
                uz_min = np.min(interpolate(velocity_z_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
                p_max = np.max(interpolate(pressure_solution, self.FenicsMesh.scalar_space_linear).vector().get_local())
                p_min = np.min(interpolate(pressure_solution, self.FenicsMesh.scalar_space_linear).vector().get_local())
                # print("X-Velocity Range = [%G, %G]" % (ux_min, ux_max))
                # print("Z-Velocity Range = [%G, %G]" % (uz_min, uz_max))
                # print("Pressure Range = [%G, %G]" % (p_min, p_max))

                # print("Norm is " + str(float(norm)) + ".")

            dt_old = copy.copy(dt_cfl)

            print("New time step is " + str(dt_cfl) + " and old time step is " + str(dt_old) + ".")

        return dt_cfl

    def solve_smb(self):

        boundary_mesh = BoundaryMesh(self.FenicsMesh.mesh, 'exterior')
        scalar_space = FunctionSpace(boundary_mesh, 'CG', self.FenicsMesh.degree)
        vector_space = VectorFunctionSpace(boundary_mesh, 'CG', self.FenicsMesh.degree)

        trial_function = TrialFunction(scalar_space)
        test_function = TestFunction(scalar_space)

        z_hat = interpolate(Constant((0.0, 1.0)), vector_space)

        solution = Function(scalar_space)
        variational_form = inner(trial_function, test_function)*dx - inner(inner(self.smb_expression, z_hat), test_function)*dx
        solve(lhs(variational_form) == rhs(variational_form), solution)

        full_solution = Function(self.FenicsMesh.scalar_space_linear)
        full_solution_array = full_solution.vector().get_local()

        dofb_vb = np.array(dof_to_vertex_map(scalar_space), dtype=int)
        vb_v = boundary_mesh.entity_map(0).array()
        v_dof = np.array(vertex_to_dof_map(self.FenicsMesh.scalar_space_linear), dtype=int)

        full_solution_array[v_dof[vb_v[dofb_vb]]] = solution.vector().get_local()
        full_solution.vector()[:] = full_solution_array

        File("smb_solution_z.pvd") << full_solution
        sol = full_solution.compute_vertex_values()

        return sol

    def get_facet_normal(self, bmesh):
        '''Manually calculate FacetNormal function'''

        if not bmesh.topology().dim() == 1:
            raise ValueError('Only works for 2-D mesh, 1-D boundary mesh.')

        vertices = bmesh.coordinates()
        cells = bmesh.cells()

        vec1 = vertices[cells[:, 1]] - vertices[cells[:, 0]]
        normals = vec1[:,[1,0]]*np.array([1,-1])
        normals /= np.sqrt((normals**2).sum(axis=1))[:, np.newaxis]

        #Ensure outward pointing normal
        [x, z] = self.FenicsMesh.get_xz_boundary_points()
        max_length = max(x)
        bmesh.init_cell_orientations(Expression(('x[0]-(L-1000)', 'x[1]'), L=max_length, degree=0))
        orientations = bmesh.cell_orientations()
        for j in range(len(orientations)):
            if orientations[j] == 0: #If inward, flip outward. Inward, 0; Outward 1
                normals[j][0] *= -1
                normals[j][1] *= -1

        V = VectorFunctionSpace(bmesh, 'DG', 0)
        V2 = VectorFunctionSpace(bmesh, 'DG', 1)

        norm = Function(V)
        nv = norm.vector()
        for n in (0,1):
            dofmap = V.sub(n).dofmap()
            for i in range(dofmap.global_dimension()):
                dof_indices = dofmap.cell_dofs(i)
                assert len(dof_indices) == 1
                nv[dof_indices[0]] = normals[i, n]

        norm2 = Function(V)
        nv2 = norm2.vector()
        for n in (0,1):
            dofmap = V.sub(n).dofmap()
            N = dofmap.global_dimension()
            for i in range(dofmap.global_dimension()):
                dof_indices = dofmap.cell_dofs(i)
                assert len(dof_indices) == 1
                nv2[dof_indices[0]] = (normals[i, n]+normals[(i-1)%N, n]+normals[(i+1)%N, n])/3

        norm3 = Function(V)
        nv3 = norm3.vector()
        for n in (0,1):
            dofmap = V.sub(n).dofmap()
            N = dofmap.global_dimension()
            for i in range(dofmap.global_dimension()):
                dof_indices = dofmap.cell_dofs(i)
                assert len(dof_indices) == 1
                nv3[dof_indices[0]] = normals[i, n] + (normals[i, n]-normals[(i-1)%N, n])

        File("normal_new.pvd") << norm3

        S = FunctionSpace(bmesh, 'DG', 0)

        normX = Function(S)
        nvX = normX.vector()
        dofmap = S.dofmap()
        for i in range(dofmap.global_dimension()):
            dof_indices = dofmap.cell_dofs(i)
            assert len(dof_indices) == 1
            nvX[dof_indices[0]] = normals[i, 0]

        normZ = Function(S)
        nvZ = normZ.vector()
        dofmap = S.dofmap()
        for i in range(dofmap.global_dimension()):
            dof_indices = dofmap.cell_dofs(i)
            assert len(dof_indices) == 1
            nvZ[dof_indices[0]] = normals[i, 1]

        normX2 = Function(S)
        nvX2 = normX2.vector()
        dofmap = S.dofmap()
        N = dofmap.global_dimension()
        for i in range(dofmap.global_dimension()):
            dof_indices = dofmap.cell_dofs(i)
            assert len(dof_indices) == 1
            nvX2[dof_indices[0]] = (normals[i, 0]+normals[(i-1)%N, 0]+normals[(i+1)%N, 0])/3

        normZ2 = Function(S)
        nvZ2 = normZ2.vector()
        dofmap = S.dofmap()
        N = dofmap.global_dimension()
        for i in range(dofmap.global_dimension()):
            dof_indices = dofmap.cell_dofs(i)
            assert len(dof_indices) == 1
            nvZ2[dof_indices[0]] = (normals[i, 1]+normals[(i-1)%N, 1]+normals[(i+1)%N, 1])/3

        norm.rename('n', 'n')
        norm2.rename('n2', 'n2')
        normX.rename('nx', 'nx')
        normZ.rename('nz', 'nz')

        self.normalvec_file << (norm, self.time)
        self.normalvec2_file << (norm2, self.time)
        self.normalx_file <<  (normX, self.time)
        self.normalz_file <<  (normZ, self.time)

        return norm, normX, normZ, normX2, normZ2

    def solve_smb_new(self, dt):
        boundary_mesh = BoundaryMesh(self.FenicsMesh.mesh, 'exterior')
        scalar_space_dg = FunctionSpace(boundary_mesh, 'DG', 0)
        scalar_space = FunctionSpace(boundary_mesh, 'CG', 1)

        norm, normX, normZ, normX2, normZ2 = self.get_facet_normal(boundary_mesh)

        a = Expression("a*nz", a=self.smb_expression_z, nz=normZ, degree=0)
        a_interp = project(a, scalar_space_dg)
        a_v = a_interp.compute_vertex_values()

        mx = Expression("m*nx", m=self.melt_expression, nx=normX, degree=0)
        mx_interp = project(mx, scalar_space_dg)
        mx_v = mx_interp.compute_vertex_values()

        mz = Expression("m*nz", m=self.melt_expression, nz=normZ, degree=1)
        mz_interp = project(mz, scalar_space)
        mz_v = mz_interp.compute_vertex_values()

        a_interp.rename('a', 'a')
        self.accumulation_file << (a_interp, self.time)
        mx_interp.rename('mx', 'mx')
        self.meltx_file << (mx_interp, self.time)
        mz_interp.rename('mz', 'mz')
        self.meltz_file << (mz_interp, self.time)

        boundary_mesh.coordinates()[:, 0] = boundary_mesh.coordinates()[:, 0] - mx_v*dt
        boundary_mesh.coordinates()[:, 1] = boundary_mesh.coordinates()[:, 1] + a_v*dt - mz_v*dt

        return boundary_mesh

    def solve_smb_numpy(self):
        boundary_mesh = BoundaryMesh(self.FenicsMesh.mesh, 'exterior')
        scalar_space = FunctionSpace(boundary_mesh, 'CG', self.FenicsMesh.degree)

        [x, z] = self.FenicsMesh.get_xz_boundary_points()

        m = np.gradient(z, x)
        tz = m
        tx = np.ones(len(m))
        tx_norm = np.divide(tx, np.sqrt(np.add(np.square(tx), np.square(tz))))
        nz = tx_norm

        scaling_factor = nz
        not_allow_smb_slope = ~np.greater(nz, np.zeros(len(nz)))
        not_allow_smb_water = np.less_equal(z, np.zeros(len(z)))
        not_allow_smb_bed = np.less_equal(z, self.FenicsMesh.bed_interpolator(x)+self.FenicsMesh.boundary_tolerance)
        scaling_factor[not_allow_smb_slope] = 0.0
        scaling_factor[not_allow_smb_water] = 0.0
        scaling_factor[not_allow_smb_bed] = 0.0

        smb_adjusted = np.multiply(self.FenicsMesh.smb_interpolator(x), scaling_factor)

        full_solution = Function(self.FenicsMesh.scalar_space_linear)
        full_solution_array = full_solution.vector().get_local()

        vb_v = boundary_mesh.entity_map(0).array()
        v_dof = np.array(vertex_to_dof_map(self.FenicsMesh.scalar_space_linear), dtype=int)

        full_solution_array[v_dof[vb_v]] = smb_adjusted
        full_solution.vector()[:] = full_solution_array

        File("numpy_smb.pvd") << full_solution
        sol = full_solution.compute_vertex_values()

        return sol

    def solve_smb_numpy_new(self, dt):

        def smb(x, z):
            if z <= 0.0 or z <= (self.FenicsMesh.bed_interpolator(x) + self.FenicsMesh.boundary_tolerance):
                return 0.0
            else:
                return FenicsMesh.smb_interpolator(x)

        def melt(x, z):
            if z > 0.0 or z <= (self.FenicsMesh.bed_interpolator(x) + self.FenicsMesh.boundary_tolerance):
                return 0.0
            else:
                return self.avg_melt


        boundary_mesh = BoundaryMesh(self.FenicsMesh.mesh, 'exterior')

        x = boundary_mesh.coordinates()[:, 0]
        z = boundary_mesh.coordinates()[:, 1]

        m = np.gradient(z, x)
        tz = m
        tx = np.ones(len(m))
        tz_norm = np.divide(tz, np.sqrt(np.add(np.square(tx), np.square(tz))))
        nx = -tz_norm
        tx_norm = np.divide(tx, np.sqrt(np.add(np.square(tx), np.square(tz))))
        nz = tx_norm

        sc = plt.scatter(x, z, c=nx)
        plt.colorbar(sc)
        plt.savefig('nx_numpy.png')
        plt.close()
        sc = plt.scatter(x, z, c=nz)
        plt.colorbar(sc)
        plt.savefig('nz_numpy.png')
        plt.close()

        return False

    def run_verification(self):

        self.compute_verification_functions()
        self.FenicsMesh.define_boundary_verification(self.analytic_velocity_x, self.analytic_velocity_z, self.analytic_pressure)

        trial_functions = TrialFunction(self.FenicsMesh.mixed_space)
        test_functions = TestFunction(self.FenicsMesh.mixed_space)
        (p, u) = split(trial_functions)
        (q, v) = split(test_functions)

        norm = 1.0
        while float(norm) >= self.solver_tolerance:
            iteration_viscosity = self.Tracers.compute_viscosity(self.iteration_velocity, self.FenicsMesh.scalar_space_dg, self.FenicsMesh.tensor_space_dg, self.FenicsMesh.degree)
            
            variational_form = 2*inner(iteration_viscosity*strain_rate(u), strain_rate(v))*dx - self.b_function(v, p) + self.b_function(u, q) - inner(self.analytic_forcing, v)*dx

            self.fenics_solution(variational_form)

            pressure_solution, velocity_solution = self.solver_solution.split()
            velocity_x_solution, velocity_z_solution = velocity_solution.split()

            velocity_solution_change = velocity_solution - self.iteration_velocity

            norm = self.norm_calculator(velocity_solution_change)/self.norm_calculator(velocity_solution)

            #Update velocity for next iteration
            assign(self.iteration_velocity, velocity_solution)

            #Print out progress
            ux_max = np.max(interpolate(velocity_x_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
            ux_min = np.min(interpolate(velocity_x_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
            uz_max = np.max(interpolate(velocity_z_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
            uz_min = np.min(interpolate(velocity_z_solution, self.FenicsMesh.scalar_space_quadratic).vector().get_local())
            p_max = np.max(interpolate(pressure_solution, self.FenicsMesh.scalar_space_linear).vector().get_local())
            p_min = np.min(interpolate(pressure_solution, self.FenicsMesh.scalar_space_linear).vector().get_local())
            # print("X-Velocity Range = [%G, %G]" % (ux_min, ux_max))
            # print("Z-Velocity Range = [%G, %G]" % (uz_min, uz_max))
            # print("Pressure Range = [%G, %G]" % (p_min, p_max))

            # print("Norm is " + str(float(norm)) + ".")

        self.save_solution()
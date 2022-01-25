from __future__ import division
from dolfin import  *
from exceptions import *

def strain_rate(velocity):
    return 0.5*(nabla_grad(velocity) + nabla_grad(velocity).T)

class ViscosityCalculator:
    def __init__(self, viscosity_type, creep_parameter, newton_viscosity, epsilon, yield_width):
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

        self.yield_width = yield_width

    def compute_viscosity(self, velocity, scalar_space, tensor_space, degree, particles, mesh):
        return self.viscosity_functions[self.viscosity_type](velocity, scalar_space, tensor_space, degree, particles, mesh)

    def calculate_glen_viscosity(self, velocity, scalar_space, tensor_space, degree, particles, mesh):
        if not self.creep_parameter:
            raise InputError(self.creep_parameter)
        viscosity_coefficient = 0.5*self.creep_parameter_exponentiated
        strainrate = strain_rate(velocity)
        viscosity = viscosity_coefficient/((0.5*strainrate[0,0]**2 + strainrate[0,1]**2 + 0.5*strainrate[1,1]**2 + self.epsilon)**(1/3))
        return viscosity

    def calculate_newton_viscosity(self, velocity, scalar_space, tensor_space, degree, particles, mesh):
        if not newton_viscosity:
            raise InputError(newton_viscosity)

        viscosity = Expression("viscosity", viscosity=self.newton_viscosity, degree=degree)
        return viscosity

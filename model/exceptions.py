from __future__ import division

class InputError(Exception):
    message = "Invalid input to simulation."
    def __init__(self, variable):
        self.variable = variable
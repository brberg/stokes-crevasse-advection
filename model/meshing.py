from __future__ import division
import os
import numpy as np
import scipy.interpolate as interpsci
from scipy.signal import argrelextrema
from dolfin import  *
from intersection import *
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

class FenicsMesh:
    def __init__(self, cell_size, hires_cell_size, steady_state=False, steady_state_location=False, inflow_velocity = 0.0, boundary_tolerance=DOLFIN_EPS, degree=1, mu=0.0, alpha=1, KIc=0.05E6, c=0.05, coulomb_strength=450000, set_coulomb_strength=False, old_meshing=False):
        self.cell_size = cell_size
        self.hires_cell_size = hires_cell_size
        self.boundary_tolerance = boundary_tolerance
        self.degree = degree

        self.bed_interpolator = interpsci.interp1d(np.linspace(0, 1E10, 100), -1E10*np.ones(100), fill_value='extrapolate')
        self.smb_interpolator = interpsci.interp1d(np.linspace(0, 1E10, 100), np.zeros(100), fill_value='extrapolate')

        self.steady_state = steady_state
        self.steady_state_location = steady_state_location

        self.inflow_velocity = inflow_velocity

        #Setup files for saving boundary
        self.boundary_file = File("boundary_timeseries.pvd", "compressed")
        self.mesh_number = 0.0

        #Calving variables
        self.mu = mu
        self.alpha = alpha
        self.KIc = KIc
        self.c = c
        self.R21c = (1-mu)/(1+mu)

        #Manual coulomb strength stuff
        self.coulomb_strength = coulomb_strength
        self.set_coulomb_strength = set_coulomb_strength

        self.old_meshing = old_meshing

    def set_bed_location(self, bed_points_x, bed_points_z):
        self.bed_interpolator = interpsci.interp1d(bed_points_x, bed_points_z, fill_value='extrapolate')

        bed_interpolator = self.bed_interpolator
        class IceThickness(UserExpression):
            def eval(self, value, x):
                value[0] = x[1] - bed_interpolator(x[0])

        self.ice_thickness_expression = IceThickness(degree=self.degree)

        boundary_tolerance = self.boundary_tolerance
        class NormalReference(UserExpression):
            def eval(self, value, x):
                if x[0] < 1000:
                    value[0] = x[0] - 500
                else:
                    value[0] = x[0] + 500
                if (x[1] <= (float(bed_interpolator(x[0])) + boundary_tolerance)) or x[1] <= 0:
                    value[1] = x[1] - 10
                else:
                    value[1] = x[1] + 10
            def value_shape(self):
                return (2,)

        self.normal_expression = NormalReference(degree=self.degree)

    def set_smb(self, smb_points_x, smb):
        self.smb_interpolator = interpsci.interp1d(smb_points_x, smb, fill_value='extrapolate')
        self.max_smb = max(smb)

    def set_steady_state_calving(self):
        [top_boundary_x, top_boundary_z], [bottom_boundary_x, bottom_boundary_z] = self.get_remeshing_boundaries()

        bottom_z_interpolator = interpsci.interp1d(bottom_boundary_x, bottom_boundary_z, fill_value='extrapolate')
        surface_z_interpolator = interpsci.interp1d(top_boundary_x, top_boundary_z, fill_value='extrapolate')
        steady_state_bottom = bottom_z_interpolator(self.steady_state_location)
        steady_state_top = surface_z_interpolator(self.steady_state_location)

        return [True, [self.steady_state_location, steady_state_bottom, steady_state_top]]

    def remove_duplicates(self, array_x, array_z, calving_location):
        array_unique_x = []
        array_unique_z = []
        cf_idx = []
        cf_z = []
        m = 0
        for j in range(len(array_x)):
            if array_x[j] in array_unique_x and array_z[j] in array_unique_z:
                pass
            else:
                array_unique_x.append(array_x[j])
                array_unique_z.append(array_z[j])
                if array_x[j] == calving_location:
                    cf_idx.append(m)
                    cf_z.append(array_z[j])
                m+=1

        keep_idx = []
        if len(cf_idx) > 0:
            keep_idx = [np.argwhere(array_unique_z==np.amax(cf_z))[0][0], np.argwhere(array_unique_z==np.amin(cf_z))[0][0]]

        delete_idx = []
        for idx in cf_idx:
            if idx in keep_idx:
                pass
            else:
                delete_idx.append(idx)
        array_unique_x = np.delete(array_unique_x, delete_idx)
        array_unique_z = np.delete(array_unique_z, delete_idx)
        return array_unique_x, array_unique_z

    def grounding_line_location(self):
        [top_boundary_x, top_boundary_z], [bottom_boundary_x, bottom_boundary_z] = self.get_remeshing_boundaries()
        surface_z_interpolator = interpsci.interp1d(top_boundary_x, top_boundary_z, fill_value='extrapolate')
        bottom_z_interpolator = interpsci.interp1d(bottom_boundary_x, bottom_boundary_z, fill_value='extrapolate')

        gl_position = 0.0
        for x in bottom_boundary_x:
            if bottom_z_interpolator(x) <= (self.bed_interpolator(x) + self.boundary_tolerance) and x > gl_position:
                gl_position = x

        if gl_position == 0:
            gl_position = max(max(top_boundary_x), max(bottom_boundary_x))

        return gl_position

    def get_hires_distance(self):
        [top_boundary_x, top_boundary_z], [bottom_boundary_x, bottom_boundary_z] = self.get_remeshing_boundaries()
        surface_z_interpolator = interpsci.interp1d(top_boundary_x, top_boundary_z, fill_value='extrapolate')
        bottom_z_interpolator = interpsci.interp1d(bottom_boundary_x, bottom_boundary_z, fill_value='extrapolate')

        gl_position = 0.0
        for x in bottom_boundary_x:
            if bottom_z_interpolator(x) <= (self.bed_interpolator(x) + self.boundary_tolerance) and x > gl_position:
                gl_position = x

        if gl_position == 0:
            gl_position = max(max(top_boundary_x), max(bottom_boundary_x))

        checking_array = np.linspace(0.0, max(bottom_boundary_x), int(max(bottom_boundary_x)/100))
        ice_thickness_array = surface_z_interpolator(checking_array) - bottom_z_interpolator(checking_array)
        ice_thickness = max(ice_thickness_array) #surface_z_interpolator(gl_position) - bottom_z_interpolator(gl_position)
        hires_distance = gl_position-5*ice_thickness

        return hires_distance

    def fake_calving(self, location):
        [top_boundary_x, top_boundary_z], [bottom_boundary_x, bottom_boundary_z] = self.get_remeshing_boundaries()
        surface_z_interpolator = interpsci.interp1d(top_boundary_x, top_boundary_z, fill_value='extrapolate')
        bottom_z_interpolator = interpsci.interp1d(bottom_boundary_x, bottom_boundary_z, fill_value='extrapolate')

        surface_location = surface_z_interpolator(location)
        bottom_location = bottom_z_interpolator(location)

        fake_calving_info = [True, [location, bottom_location, surface_location]]

        self.remesh(fake_calving_info)

    def get_remeshing_boundaries(self):
        # Mark boundary vertices
        boundary_labels = []
        boundary_vertices = []
        boundary_ordered_vertices = []
        for v in vertices(self.mesh):
            for f in facets(v):
                if f.exterior():
                    boundary_vertices.append(v)
                    boundary_labels.append(v.index())
                    break

        # Ordering the boundary vertices
        bdry_no = len(boundary_vertices)
        head = boundary_vertices[0]
        boundary_ordered_vertices.append(head) # keep in mind that here head is a fenics object so head.index() is
        count = 1  # different from the boundary_labels.index(...) below
        while count < bdry_no:
            flag = 0
            id = boundary_labels.index(head.index()) # returns the lowest index in list that head.index() appears
            del boundary_labels[id]
            del boundary_vertices[id]
            for f in facets(head):
                if f.exterior():
                    for v in vertices(f):
                        if v.index() in boundary_labels:
                            boundary_ordered_vertices.append(v)
                            head = v
                            flag = 1
                            count += 1
                            break
                if flag == 1:
                    break

        top_boundary_x = []
        top_boundary_z = []
        bottom_boundary_x = []
        bottom_boundary_z = []
        x = self.mesh.coordinates()[:, 0]
        z = self.mesh.coordinates()[:, 1]
        for v in boundary_ordered_vertices:
            k = v.index()
            if x[k] != 0.0:
                if z[k] <= (self.bed_interpolator(x[k]) + self.boundary_tolerance) or z[k] <= 0.0:
                    bottom_boundary_x.append(x[k])
                    bottom_boundary_z.append(z[k])
                else:
                    top_boundary_x.append(x[k])
                    top_boundary_z.append(z[k])

        if top_boundary_x[0] > top_boundary_x[-1]:
            top_boundary_x = top_boundary_x[::-1]
            top_boundary_z = top_boundary_z[::-1]
        if bottom_boundary_x[0] > bottom_boundary_x[-1]:
            bottom_boundary_x = bottom_boundary_x[::-1]
            bottom_boundary_z = bottom_boundary_z[::-1]

        return [top_boundary_x, top_boundary_z], [bottom_boundary_x, bottom_boundary_z]

    def check_counterclockwise_ordering(self, boundary_points):
        running_total = 0
        for j in range(np.shape(boundary_points)[1]):
            if j == 0:
                k = np.shape(boundary_points)[1] - 1
            else:
                k = j -1
            calc = (boundary_points[0][j] - boundary_points[0][k])*(boundary_points[1][j]+boundary_points[1][k])
            running_total += calc

        if running_total <= 0:
            return True
        else:
            return False


    def get_xz_boundary_points(self):
        "Function to get boundary points of a mesh. Taken from Yue's code and adapted to work with newer Fenics version."
        # Mark boundary vertices
        boundary_labels = []
        boundary_vertices = []
        boundary_ordered_vertices = []
        for v in vertices(self.mesh):
            for f in facets(v):
                if f.exterior():
                    boundary_vertices.append(v)
                    boundary_labels.append(v.index())
                    break

        # Ordering the boundary vertices
        bdry_no = len(boundary_vertices)
        head = boundary_vertices[0]
        boundary_ordered_vertices.append(head) # keep in mind that here head is a fenics object so head.index() is
        count = 1  # different from the boundary_labels.index(...) below
        while count < bdry_no:
            flag = 0
            id = boundary_labels.index(head.index()) # returns the lowest index in list that head.index() appears
            del boundary_labels[id]
            del boundary_vertices[id]
            for f in facets(head):
                if f.exterior():
                    for v in vertices(f):
                        if v.index() in boundary_labels:
                            boundary_ordered_vertices.append(v)
                            head = v
                            flag = 1
                            count += 1
                            break
                if flag == 1:
                    break

        boundary_points_x = []
        boundary_points_z = []
        x = self.mesh.coordinates()[:, 0]
        z = self.mesh.coordinates()[:, 1]
        for v in boundary_ordered_vertices:
            k = v.index()
            boundary_points_x.append(x[k])
            boundary_points_z.append(z[k])
        boundary_points = [boundary_points_x, boundary_points_z]

        is_counterclockwise = self.check_counterclockwise_ordering(boundary_points)

        if is_counterclockwise:
            boundary_points_final = boundary_points
        else:
            boundary_points_final = [boundary_points_x[::-1], boundary_points_z[::-1]]

        return boundary_points_final

    def check_shear_threshold(self, shear_stress, shear_strength, tensile_stress, tensile_strength, width):
        if shear_stress >= shear_strength or tensile_stress>tensile_strength or width>0.0099:
            stress_exceeded = True
        else:
            stress_exceeded = False

        return stress_exceeded

    def increment_fault_plane(self, xp, zp, sigma00, sigma01, sigma11, polygon, tensile_strength, width, ref_angle = None, initial=False):
        s00 = sigma00(xp, zp)
        s01 = sigma01(xp, zp)
        s11 = sigma11(xp, zp)

        s_matrix = np.array([[s00, s01], [s01, s11]])
        eigenvalues, eigenvectors = np.linalg.eig(s_matrix)
        P = eigenvectors
        P_inv = np.linalg.inv(P)
        X = np.dot(P_inv, s_matrix)
        s_diag = np.diag(np.dot(X, P))

        if (np.amax(s_diag)+(zp<=0)*(1000*9.81*(0-zp))) > tensile_strength or width>0.0099:
            fault_angle = 0.0
        else:
            phi = np.arctan(self.mu)
            fault_angle = (np.pi/4-phi/2)
        angles = [(np.arctan(eigenvector[1]/eigenvector[0])+fault_angle) for eigenvector in eigenvectors]
        for j in range(len(angles)):
            while angles[j] < 0 or angles[j] >= 2*np.pi:
                if angles[j] < 0:
                    angles[j] += 2*np.pi
                if angles[j] >= 2*np.pi:
                    angles[j] -= 2*np.pi

        step_length = self.hires_cell_size

        if initial:
            increments = []
            path = mpltPath.Path(polygon)
            for angle in angles:
                x_inc = xp + step_length*np.cos(angle)
                z_inc = zp + step_length*np.sin(angle)
                inside = path.contains_points([[x_inc,z_inc]])[0]
                if inside:
                    increments.append([x_inc, z_inc, angle])
                else:
                    x_inc2 = xp + step_length*np.cos(angle+np.pi)
                    z_inc2 = zp + step_length*np.sin(angle+np.pi)
                    inside2 = path.contains_points([[z_inc2, z_inc2]])[0]
                    if inside2:
                        increments.append([x_inc2, z_inc2, angle+np.pi])
            return increments
        else:
            angles_list = [angles[0], angles[1], angles[0]+np.pi, angles[1]+np.pi]
            for j in range(len(angles_list)):
                while angles_list[j] < 0 or angles_list[j] >= 2*np.pi:
                    if angles_list[j] < 0:
                        angles_list[j] += 2*np.pi
                    if angles_list[j] >= 2*np.pi:
                        angles_list[j] -= 2*np.pi
            final_angles_list = []
            for angle in angles_list:
                if ref_angle <= np.pi and angle <= np.pi:
                    final_angles_list.append(angle)
                elif ref_angle > np.pi and angle > np.pi:
                    final_angles_list.append(angle)

            angles_diff = np.absolute(final_angles_list - ref_angle)
            angle_index = np.argmin(angles_diff)
            chosen_angle = final_angles_list[angle_index]
            x_inc = xp + step_length*np.cos(chosen_angle)
            z_inc = zp + step_length*np.sin(chosen_angle)

            path = mpltPath.Path(polygon)
            inside_boundary = path.contains_points([[x_inc,z_inc]])[0]

            return [x_inc, z_inc, chosen_angle, inside_boundary]

    def yield_stress(self, xp, zp, sigma00, sigma01, sigma11):
        np.seterr(divide='raise')
        s00 = sigma00(xp, zp)
        s01 = sigma01(xp, zp)
        s11 = sigma11(xp, zp)
        s_matrix = np.array([[s00, s01], [s01, s11]])
        eigenvalues, P = np.linalg.eig(s_matrix)
        P_inv = np.linalg.inv(P)
        X = np.dot(P_inv, s_matrix)
        s_diag = np.diag(np.dot(X, P))
        sigma2 = -np.amax(s_diag)
        sigma1 = -np.amin(s_diag)
        R21 = sigma2/sigma1
        if self.set_coulomb_strength:
            return self.coulomb_strength
        elif R21 >= self.R21c:
            return 1E10
        else:
            num = 2*self.KIc
            denom1 = ((1+(1-self.mu*(1+R21)/(1-R21))**(2/3)-1)**(1/2))
            denom2 = (1+3*self.mu**2*self.alpha**2*(1-R21)**2)**(1/2)
            strength = num/(denom1*denom2*self.c**(1/2))
            return strength

    def calculate_shear_calving_event(self, sigma00, sigma01, sigma11, tensile_strength, Tracers):
        xt, zt = np.transpose(Tracers.particles.return_property(self.mesh, 0))
        ft = Tracers.particles.return_property(self.mesh, 1)
        current_width = interpsci.NearestNDInterpolator(np.transpose([xt, zt]), ft)
        def calculate_width(x, z):
            try:
                output = current_width(x, z)
            except:
                output = 0.0
            return output

        [x, z] = self.get_xz_boundary_points()
        max_length = max(x)
        calving_x = 0.0
        calving_z = 0.0
        yield_stress_array = []
        for j in range(len(x)):
            s00 = sigma00(x[j], z[j])
            s01 = sigma01(x[j], z[j])
            s11 = sigma11(x[j], z[j])
            s_matrix = np.array([[s00, s01], [s01, s11]])
            eigenvalues, P = np.linalg.eig(s_matrix)
            P_inv = np.linalg.inv(P)
            X = np.dot(P_inv, s_matrix)
            s_diag = np.diag(np.dot(X, P))
            yield_stress_array.append(-np.amin(s_diag))

        extrema_shear_stress_indices = argrelextrema(np.asarray(yield_stress_array), np.greater)[0]

        extrema_x = [x[index] for index in extrema_shear_stress_indices]
        extrema_z = [z[index] for index in extrema_shear_stress_indices]
        extrema_shear_stress = [yield_stress_array[index] for index in extrema_shear_stress_indices]

        extrema_x_final = []
        extrema_z_final = []
        extrema_shear_stress_final = []

        for j in range(len(extrema_shear_stress)):
            if extrema_shear_stress[j] >= self.yield_stress(extrema_x[j], extrema_z[j], sigma00, sigma01, sigma11):
                extrema_shear_stress_final.append(extrema_shear_stress[j])
                extrema_x_final.append(extrema_x[j])
                extrema_z_final.append(extrema_z[j])

        plt.plot(x,z, 'black')

        stress_status = []
        x_plots = []
        z_plots = []

        for j in range(len(extrema_shear_stress_final)):

            width = calculate_width(extrema_x_final[j], extrema_z_final[j])
            increments = self.increment_fault_plane(extrema_x_final[j], extrema_z_final[j], sigma00, sigma01, sigma11, np.transpose([x, z]), tensile_strength, width, initial=True)

            for increment in increments:
                x_plot = []
                z_plot = []
                x_plot.append(extrema_x_final[j])
                z_plot.append(extrema_z_final[j])
                x_new = increment[0]
                z_new = increment[1]
                width = calculate_width(x_new, z_new)
                ref_angle = increment[2]
                stress_exceeded = True
                inside_boundary = True
                x_plot.append(x_new)
                z_plot.append(z_new)
                while stress_exceeded and inside_boundary:
                    x_new, z_new, ref_angle, inside_boundary = self.increment_fault_plane(x_new, z_new, sigma00, sigma01, sigma11, np.transpose([x, z]), tensile_strength, width, ref_angle=ref_angle)
                    width = calculate_width(x_new, z_new)
                    if inside_boundary:
                        x_plot.append(x_new)
                        z_plot.append(z_new)
                        s00 = sigma00(x_new, z_new)
                        s01 = sigma01(x_new, z_new)
                        s11 = sigma11(x_new, z_new)
                        s_matrix = np.array([[s00, s01], [s01, s11]])
                        eigenvalues, P = np.linalg.eig(s_matrix)
                        P_inv = np.linalg.inv(P)
                        X = np.dot(P_inv, s_matrix)
                        s_diag = np.diag(np.dot(X, P))
                        stress_exceeded = self.check_shear_threshold(-np.amin(s_diag), self.yield_stress(x_new, z_new, sigma00, sigma01, sigma11), np.amax(s_diag)+(z_new<=0)*(1000*9.81*(0-z_new)), tensile_strength, width)
                    else:
                        try:
                            x_line = np.append(x_plot, x_new)
                            z_line = np.append(z_plot, z_new)
                            x_intersect, z_intersect = intersection(x, z, x_line[1:], z_line[1:])
                            x_plot.append(x_intersect[0])
                            z_plot.append(z_intersect[0])
                        except:
                            print("Failed to add boundary intersection point")
            
                x_plots.append(x_plot)
                z_plots.append(z_plot)
                stress_status.append(stress_exceeded)
                plt.plot(x_plot, z_plot, ls='--', color='grey')

        possible_paths = []
        for j in range(len(stress_status)):
            if stress_status[j]:
                possible_paths.append(j)

        path_min = 1E15
        priority_index = False
        for index in possible_paths:
            surface1 = (z_plots[index][0]-self.bed_interpolator(x_plots[index][0])) >= 2*self.hires_cell_size
            surface2 = (z_plots[index][-1]-self.bed_interpolator(x_plots[index][-1])) >= 2*self.hires_cell_size
            if min(x_plots[index]) < path_min and not (surface1 and surface2) and not (not surface1 and not surface2):
                path_min = min(x_plots[index])
                priority_index = index

        if np.any(stress_status) and priority_index:
            shear_calving_event = True

            x_path = x_plots[priority_index][::-1]
            z_path = z_plots[priority_index][::-1]

            surface_event = z_path[-1] > z_path[0]

            x_temp = []
            z_temp = []

            if surface_event:
                for j in range(len(x)):
                    if x[j] == x_path[-1] and z[j] == z_path[-1]:
                        next_index = j+1
                        break
                    x_temp.append(x[j])
                    z_temp.append(z[j])
            else:
                x = x[::-1]
                z = z[::-1]
                for j in range(len(x)):
                    if x[j] == x_path[-1] and z[j] == z_path[-1]:
                        next_index = j+1
                        break
                    x_temp.append(x[j])
                    z_temp.append(z[j])

            x_final = []
            z_final = []
            
            for j in range(len(x_temp)):
                if j == 0:
                    x_is_between = False
                    z_is_between = False
                else:
                    x_is_between = (x_temp[j-1] <= x_path[0] <= x_temp[j]) or (x_temp[j-1] >= x_path[0] >= x_temp[j])
                    z_is_between = (z_temp[j-1] <= z_path[0] <= z_temp[j]) or (z_temp[j-1] >= z_path[0] >= z_temp[j])
                if x_is_between and z_is_between:
                    break
                else:
                    x_final.append(x_temp[j])
                    z_final.append(z_temp[j])

            for j in range(len(x_path)):
                x_final.append(x_path[j])
                z_final.append(z_path[j])

            for j in range(next_index, len(x)):
                x_final.append(x[j])
                z_final.append(z[j])

            if not surface_event:
                x_final = x_final[::-1]
                z_final = z_final[::-1]

            plt.plot(x_final, z_final, color='red')
            plt.xlim(max(x_final)-900, max(x_final)+100)
            plt.savefig('testcontours_calved' + str(int(self.mesh_number)) + '.png')
            plt.close()

        else:
            shear_calving_event = False

            x_final = None
            z_final = None
            plt.xlim(max(x)-900, max(x)+100)
            plt.savefig('testcontours' + str(int(self.mesh_number)) + '.png')
            plt.close()

        tracer_x_truncate_location = None
        return [x_final, z_final], tracer_x_truncate_location, shear_calving_event

    def get_shear_calving_points(self, max_shear_stress, shear_angle, shear_strength):
        shear_calving_event = False

        [x, z] = self.get_xz_boundary_points()
        max_length = max(x)
        calving_x = 0.0
        calving_z = 0.0

        for j in range(len(x)):
            if z[j] <= 0.0 and x[j] > calving_x:
                if max_shear_stress(x[j], z[j]) >= shear_strength:
                    calving_x = x[j]
                    calving_z = z[j]

        front_stress_check = (calving_x != 0.0)

        slope = np.tan(shear_angle*np.pi/180)
        calving_line = -slope*(x-calving_x)+calving_z

        xi, zi = intersection(x, z, x, calving_line)
        calving_finish_x = xi[-1]
        calving_finish_z = zi[-1]

        ending_stress_check = max_shear_stress(calving_finish_x, calving_finish_z) >= shear_strength
        intermediate_stress_check = True

        #Counterclockwise points
        plt.plot(x, z, color='red')
        plt.plot(x, calving_line, color="green")
        plt.plot(xi, zi, "*k")
        plt.xlim((48000, 49100))
        plt.ylim((-200, 200))
        x_final = []
        z_final = []
        step = 0
        for j in range(len(x)):
            if step == 0:
                if x[j] < calving_x or z[j] < calving_z:
                    x_final.append(x[j])
                    z_final.append(z[j])
                else:
                    x_final.append(xi[0])
                    z_final.append(zi[0])
                    x_final.append(xi[-1])
                    z_final.append(zi[-1])
                    step = 1
            elif step == 1:
                if x[j] >= calving_finish_x:
                    pass
                else:
                    x_final.append(x[j])
                    z_final.append(z[j])
                    step = 2
                if x[j] >= calving_finish_x and x[j] <= calving_x and intermediate_stress_check:
                    corresponding_z = -slope*(x[j]-calving_x)+calving_z
                    if max_shear_stress(x[j], corresponding_z) < shear_strength:
                        intermediate_stress_check = False
            elif step == 2:
                    x_final.append(x[j])
                    z_final.append(z[j])

        plt.scatter(x_final, z_final, color='blue')
        plt.savefig('test_shear.png')

        if front_stress_check and ending_stress_check and intermediate_stress_check:
            shear_calving_event = True

        return [x_final, z_final], calving_finish_x, shear_calving_event

    def calve_boundary_points(self, boundary_points, calving_info):
        calved_boundary_x = []
        calved_boundary_z = []
        for j in range(np.shape(boundary_points)[1]):
            if boundary_points[0][j] < (calving_info[1][0]):
                calved_boundary_x.append(boundary_points[0][j])
                calved_boundary_z.append(boundary_points[1][j])
            elif boundary_points[1][j] < calving_info[1][1]:
                calved_boundary_x.append(calving_info[1][0])
                calved_boundary_z.append(calving_info[1][1])
            elif boundary_points[1][j] > calving_info[1][2]:
                calved_boundary_x.append(calving_info[1][0])
                calved_boundary_z.append(calving_info[1][2])
            else:
                calved_boundary_x.append(calving_info[1][0])
                calved_boundary_z.append(boundary_points[1][j])

        calved_boundary_x, calved_boundary_z = self.remove_duplicates(calved_boundary_x, calved_boundary_z, calving_info[1][0])

        return [calved_boundary_x, calved_boundary_z]

    def remesh_inflow(self, inflow_bottom, inflow_top, calving_info):
        [top_boundary_x, top_boundary_z], [bottom_boundary_x, bottom_boundary_z] = self.get_remeshing_boundaries()

        [boundary_points_x, boundary_points_z] = self.get_xz_boundary_points()

        minimum_x_top_bottom = np.amin(np.concatenate((top_boundary_x, bottom_boundary_x))) 

        filtered_boundary_points_x = []
        filtered_boundary_points_z = []
        for j in range(len(boundary_points_x)):
            if boundary_points_x[j] >= (minimum_x_top_bottom + self.boundary_tolerance):
                filtered_boundary_points_x.append(boundary_points_x[j])
                filtered_boundary_points_z.append(boundary_points_z[j])

        z_differences = []
        for j in range(len(filtered_boundary_points_z)-1):
            z_differences.append(abs(filtered_boundary_points_z[j+1] - filtered_boundary_points_z[j]))
        z_differences.append(abs(filtered_boundary_points_z[0] - filtered_boundary_points_z[-1]))

        left_boundary_location = np.argmax(z_differences) + 1

        left_boundary_x = [0.0, 0.0]
        left_boundary_z = [inflow_top, inflow_bottom]

        final_boundary_x = np.concatenate((filtered_boundary_points_x[:left_boundary_location], left_boundary_x, filtered_boundary_points_x[left_boundary_location:]))
        final_boundary_z = np.concatenate((filtered_boundary_points_z[:left_boundary_location], left_boundary_z, filtered_boundary_points_z[left_boundary_location:]))

        boundary_points = [final_boundary_x, final_boundary_z]

        if calving_info[0]:
            boundary_points = self.calve_boundary_points(boundary_points, calving_info)

        self.make_gmsh_unstructured(boundary_points)

    def remesh(self, calving_info):
        boundary_points = self.get_xz_boundary_points()
        if calving_info[0]:
            boundary_points = self.calve_boundary_points(boundary_points, calving_info)

        if self.old_meshing:
            self.make_gmsh_unstructured_old(boundary_points)
        else:
            distance_check = self.get_hires_distance()
            if distance_check >= 1000:
                try:
                    self.make_gmsh_unstructured(boundary_points)
                except:
                    self.make_gmsh_unstructured_old(boundary_points)
            else:
                self.make_gmsh_unstructured_old(boundary_points)


    def remesh_old(self, calving_info):
        boundary_points = self.get_xz_boundary_points()
        if calving_info[0]:
            boundary_points = self.calve_boundary_points(boundary_points, calving_info)

        self.make_gmsh_unstructured_old(boundary_points)

    def shear_remesh(self, boundary_points):
        distance_check = self.get_hires_distance()
        if distance_check >= 1000:
            try:
                self.make_gmsh_unstructured(boundary_points)
            except:
                print("Try except failed. Using old meshing.")
                self.make_gmsh_unstructured_old(boundary_points)
        else:
            print("Invalid distance check. Using old meshing.")
            self.make_gmsh_unstructured_old(boundary_points)

    def load_mesh(self, filename):
        self.mesh = Mesh(filename)
        self.define_function_spaces()
        self.define_boundaries()
        self.remesh_old(calving_info=[False, [1E15, 0.0, 0.0]])

    def make_gmsh_unstructured_old(self, xz_boundary):
        "Makes gmsh using gmsh without removing any points and using a single given cell size for the unstructured mesh."
        x = xz_boundary[0]
        z = xz_boundary[1]
        num_points = np.shape(x)[0]
        try:
            hires_distance = self.get_hires_distance()
            print("Hires distance is " + str(hires_distance) + ".")
        except:
            print("Failed getting grounding line location for variable resolution. Proceeding with low resolution only.")
            hires_distance = 1E10

        if hires_distance <= 1000:
            hires_distance = 1E10

        geo_file = open('new-mesh.geo', 'w')

        point_counter = 0
        for j in range(0, num_points):
            geo_file.write("point_list[%g] = newp; \n"%point_counter)
            if x[j] >= hires_distance:
                cell_size = self.hires_cell_size
            else:
                cell_size = self.cell_size
            geo_file.write("Point(point_list[%g]) = {%1.13f, %1.13f, 0, %g}; \n"%(point_counter, x[j], z[j], cell_size))
            point_counter += 1

        line_list = []
        line_counter = 1
        line_list.append(line_counter)
        for j in range(point_counter-1):
            geo_file.write("Line(%g) = {point_list[%g], point_list[%g]}; \n"%(line_counter, line_counter-1, line_counter))
            line_counter += 1
            line_list.append(line_counter)
        geo_file.write("Line(%g) = {point_list[%g], point_list[0]}; \n"%(line_counter, line_counter-1))
        line_counter += 1

        line_list_string = ""
        line_list_string += str(line_list[0])
        for j in range (1, len(line_list)):
            line_list_string += ", " + str(line_list[j])

        geo_file.write("Line Loop(%g) = {%s}; \n"%(line_counter, line_list_string))
        line_loop_index = line_counter
        line_counter += 1
        plane_surface_index = line_counter
        geo_file.write("Plane Surface(%g) = {%g}; \n"%(plane_surface_index, line_loop_index))

        geo_file.close()

        os.system("/Applications/Gmsh.app/Contents/MacOS/Gmsh new-mesh.geo -2")
        os.system("dolfin-convert new-mesh.msh new-mesh.xml")
        self.mesh = Mesh("new-mesh.xml")

        [os.remove("new-mesh" + extension) for extension in [".geo", ".msh", ".xml"]]

        self.define_function_spaces()
        self.define_boundaries()

    def make_gmsh_unstructured(self, xz_boundary):
        "Makes gmsh using gmsh without removing any points and using a single given cell size for the unstructured mesh."
        x = xz_boundary[0]
        z = xz_boundary[1]
        num_points = np.shape(x)[0]
        try:
            hires_distance = self.get_hires_distance()
            print("Hires distance is " + str(hires_distance) + ".")
        except:
            print("Failed getting grounding line location for variable resolution. Proceeding with low resolution only.")
            hires_distance = 1E10

        if hires_distance <= 1000:
            hires_distance = 1E10

        geo_file = open('new-mesh.geo', 'w')

        point_counter = 0
        lores_counter = 0
        lores_x = []
        need_jval = True
        for j in range(0, num_points):
            if x[j] >= hires_distance:
                if need_jval:
                    jval = j
                    need_jval = False
                geo_file.write("point_list[%g] = newp; \n"%point_counter)
                cell_size = self.hires_cell_size
                geo_file.write("Point(point_list[%g]) = {%1.13f, %1.13f, 0, %g}; \n"%(point_counter, x[j], z[j], cell_size))
                point_counter += 1
            else:
                geo_file.write("lowres_list[%g] = newp; \n"%lores_counter)
                cell_size = self.cell_size
                geo_file.write("Point(lowres_list[%g]) = {%1.13f, %1.13f, 0, %g}; \n"%(lores_counter, x[j], z[j], cell_size))
                lores_counter += 1
                lores_x.append(x[j])

        line_list = []
        line_counter = 1
        line_list.append(line_counter)
        ####
        for j in range(jval-1):
            geo_file.write("Line(%g) = {lowres_list[%g], lowres_list[%g]}; \n"%(line_counter, line_counter-1, line_counter))
            line_counter += 1
            line_list.append(line_counter)

        geo_file.write("Line(%g) = {lowres_list[%g], point_list[0]}; \n"%(line_counter, line_counter-1))
        line_counter += 1
        line_list.append(line_counter)

        geo_file.write("Spline(newl) = point_list[]; \n")
        line_counter += 1
        line_list.append(line_counter)

        geo_file.write("Line(%g) = {point_list[%g], lowres_list[%g]}; \n"%(line_counter, point_counter-1, jval+1))
        line_counter += 1
        line_list.append(line_counter)


        for j in range(jval, lores_counter-2):
            geo_file.write("Line(%g) = {lowres_list[%g], lowres_list[%g]}; \n"%(line_counter, line_counter-2, line_counter-1))
            line_counter += 1
            line_list.append(line_counter)

        geo_file.write("Line(%g) = {lowres_list[%g], lowres_list[0]}; \n"%(line_counter, line_counter-2))
        line_counter += 1

        line_list_string = ""
        line_list_string += str(line_list[0])
        for j in range (1, len(line_list)):
            line_list_string += ", " + str(line_list[j])

        geo_file.write("Line Loop(%g) = {%s}; \n"%(line_counter, line_list_string))
        line_loop_index = line_counter
        line_counter += 1
        plane_surface_index = line_counter
        geo_file.write("Plane Surface(%g) = {%g}; \n"%(plane_surface_index, line_loop_index))

        geo_file.close()

        os.system("/Applications/Gmsh.app/Contents/MacOS/Gmsh new-mesh.geo -2")
        os.system("dolfin-convert new-mesh.msh new-mesh.xml")
        self.mesh = Mesh("new-mesh.xml")

        self.define_function_spaces()
        self.define_boundaries()

    def define_function_spaces(self):
        self.scalar_element = FiniteElement("CG", self.mesh.ufl_cell(), self.degree)
        self.vector_element = VectorElement("CG", self.mesh.ufl_cell(), self.degree+1)
        self.scalar_space_linear = FunctionSpace(self.mesh, "CG", self.degree)
        self.scalar_space_quadratic = FunctionSpace(self.mesh, "CG", self.degree+1)
        self.vector_space_linear = VectorFunctionSpace(self.mesh, "CG", self.degree)
        self.vector_space_quadratic = VectorFunctionSpace(self.mesh, "CG", self.degree+1)
        self.mixed_space = FunctionSpace(self.mesh, MixedElement([self.scalar_element, self.vector_element]))
        self.tensor_space = TensorFunctionSpace(self.mesh, "CG", self.degree)
        self.scalar_space_dg = FunctionSpace(self.mesh, "DG", self.degree)
        self.vector_space_dg = VectorFunctionSpace(self.mesh, "DG", self.degree)
        self.tensor_space_dg = TensorFunctionSpace(self.mesh, "DG", self.degree)

    def define_boundaries(self):
        boundary_tolerance = self.boundary_tolerance
        bed_interpolator = self.bed_interpolator

        boundary_meshfunction = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1, 0)
        boundary_meshfunction.set_all(0)

        class Boundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        Boundary().mark(boundary_meshfunction, 5)

        class Grounded(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (x[1] <= (float(bed_interpolator(x[0])) + boundary_tolerance))

        Grounded().mark(boundary_meshfunction, 1)

        class Underwater(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] <= 0
        
        Underwater().mark(boundary_meshfunction, 2)

        class GroundedUnderwater(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (x[1] <= (float(bed_interpolator(x[0])) + boundary_tolerance)) and x[1] <= 0

        GroundedUnderwater().mark(boundary_meshfunction, 3)

        class Upstream(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 0, boundary_tolerance)

        Upstream().mark(boundary_meshfunction, 4)

        bc_velocity = DirichletBC(self.mixed_space.sub(1).sub(0), Constant(self.inflow_velocity), boundary_meshfunction, 4)

        self.boundary_conditions = [bc_velocity]
        self.ds = Measure("ds")(subdomain_data=boundary_meshfunction)

        boundary_meshfunction.rename('boundary', 'boundary')
        self.boundary_file << (boundary_meshfunction, self.mesh_number)
        self.mesh_number += 1

    def define_boundary_verification(self, analytic_velocity_x, analytic_velocity_z, analytic_pressure):
        boundary_tolerance = self.boundary_tolerance
        cell_size = self.cell_size
        
        velocity_boundary_meshfunction = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1, 0)
        velocity_boundary_meshfunction.set_all(0)
        pressure_boundary_meshfunction = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1, 0)
        pressure_boundary_meshfunction.set_all(0)

        class Origin(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0, boundary_tolerance) and near(x[1], 0, 2*cell_size) and on_boundary
        
        Origin().mark(pressure_boundary_meshfunction, 1)

        class FullBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        FullBoundary().mark(velocity_boundary_meshfunction, 1)

        bc_velocity_x = DirichletBC(self.mixed_space.sub(1).sub(0), analytic_velocity_x, velocity_boundary_meshfunction, 1)
        bc_velocity_z = DirichletBC(self.mixed_space.sub(1).sub(1), analytic_velocity_z, velocity_boundary_meshfunction, 1)
        bc_pressure = DirichletBC(self.mixed_space.sub(0), analytic_pressure, pressure_boundary_meshfunction, 1)

        self.boundary_conditions = [bc_velocity_x, bc_velocity_z, bc_pressure]

        File("velocity_boundary_meshfunction.pvd") << velocity_boundary_meshfunction
        File("pressure_boundary_meshfunction.pvd") << pressure_boundary_meshfunction

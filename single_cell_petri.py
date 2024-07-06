from __future__ import print_function
# import necessary packages
import pandas as pd
import numpy as np
from array import *
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
# from matplotlib.animation import FuncAnimation
from pylab import *
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from copy import *
import csv
from collections import Counter

import shapely.geometry as geom #import Point
from shapely.ops import cascaded_union
from itertools import combinations

import gc

#import my own scripts
from Line import *    #line coefficient, foot point on a line

# import FEniCS packages
from fenics import *
from mshr import *

from mpi4py import MPI
import time as TM
import dolfinx
from dolfinx import *
import ufl

"""File Description:
This file is for multiple cells with shape changing during movement. The orientation of the cell is included. 
At some time point, the high concentration of signalling molecules triggers the differentiation of the cell, therefore,
the equilibrium cell phenotype could change.

Domain: Computational domain (container) + wound domain (rectangle)
Mesh: Moving mesh, which is approximated by Jacobian (i.e. everything is solved in the original mesh)
Possible Cell Shape: 'circle' -> 'ellipse'
Cell Displacement: - chemotaxis
                   - random walk
                   - cell equilibrium
                   - passive convection (morphoelasticity)
Initial settings: - u_TGbeta_0=0
                  - u_PDGF_0=1 in wound & u_PDGF_0=0 otherwise
                  - u_mechan=(0,0)

"""
np.random.seed(10)

### get the cell index that the point is in when it is not nodal points or on the line segment
def cell_collisions(point):
    collisions=mesh.bounding_box_tree().compute_collisions(Point(point[0],point[1]))
    n=np.shape(collisions)[0]
    cell_index=[]
    for i in range(0,n):
        if Cell(mesh,collisions[i]).contains(Point(point[0],point[1])):
            cell_index=cell_index+[collisions[i]]
    return cell_index  # the shape might be 1 (inside the mesh cell), 2 (on the edge) and more (at the nodal point)


def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def create_boundary_mesh(coordinates):
    x_coords = coordinates[0]
    y_coords = coordinates[1]

    # Create list of Points from x_coords and y_coords
    points_list = [Point(x, y) for x, y in zip(x_coords, y_coords)]
    # Create Polygon from the list of points
    polygon = Polygon(points_list)
    # Generate the mesh
    mesh = generate_mesh(polygon, 10)

    boundarymesh = BoundaryMesh(mesh, "exterior", True)

    return boundarymesh

##########################################################
## Parameters' Value
##########################################################
Es=100 # Young's modulus of ECM
Ew = 300 # Young's modulus of the scar
Ec=5 # Young's modulus of cell
F=10 # cell mutual tractionforce
R=5

nu=0.48 #Poisson's ratio
P=2.08*17 # magnitude of body force in mechanics part
Q_0=3.3*20 # magnitude of plastic force
rho=1E4 # fibrin and collagen constant
rho_v=2.5 # speed of biased movement for the cells
rho_morpho=1

mu_TGbeta=7

##########################################################
## settings of cell moving
##########################################################
## computational domain
# x0=60
# y0=60
# #vetices of the container
# boundpoints=np.array([[-x0,x0,x0,-x0],[-y0,-y0,y0,y0]]) #vertices of the polygon container

# wound_fct_express=Expression('1/(2*2)*(1+tanh((wx-x[0])/2))*(1+tanh((wx+x[0])/2))*1/(2*2)*(1+tanh((wy-x[1])/2))*(1+tanh((wy+x[1])/2))',degree=2, wx=wx,wy=wy)
#wound_fct_express=Expression('-wx<=x[0] && wx>=x[0] && -wy<=x[1] && wy>=x[1]? 1.0:0.0', degree=2, wx=wx,wy=wy)
##########################################################
## Random Walk parameters#######
##########################################################
mean = [0,0] # mean of rw
cov = [[1,0], [0,1]] # covariance of rw
sigma_rw=3 # size of rw

dt=0.05 #time

##########################################################
## Other parameters#######
##########################################################
sigma_concent=20 # concentration weight for differentiation
plot_phi = np.arange(0, 2*np.pi, 0.01) # angles to plot circle


#### set initial position of cells
#r_cell=np.array([[50,50,-50,-50],[30,-30,-30,30]])

# r_cell = np.array([[20,20,-20,-20, 10, -10, -10, 10],[20,-20,-20,20, -10, 10, -10,10]])
r_cell = np.array([[20],[20]])
#r_cell = np.array([[10, -10, -10, 10],[-10, 10, -10,10]])
r_cell_plot = r_cell

r_cell_disp = np.zeros(np.shape(r_cell))  # normal fibroblasts positions

# mechanics: angles to simulate the point force
poly_type='inscribed' # the type of the polygon, either 'inscribed' or 'eqarea'
poly_n=30 # number of edges of the polygon
phi_cell=np.random.uniform(0,2*np.pi,r_cell.shape[1])

ell_a = 6.25
ell_b = 4

hypo_b = 1/sqrt(6)*R
hypo_a = 4 * hypo_b


lam_s = 10



beta_one = 0.2
beta_two = 0.001
lambda_curr = dict()

alpha = 2



tau = dict()
for i in range(r_cell.shape[1]):
    tau[i] = np.zeros((poly_n, 2))
    lambda_curr[i] = 0.5

cell_centroid = dict()
for i in range(r_cell.shape[1]):
    cell_centroid[i] = np.array([]).reshape((2,-1))
    cell_centroid[i] = np.append(cell_centroid[i], r_cell[:,i][:, newaxis], axis=1)


#boundary is x = 60, y = 60, x = -60, y = -60 resptively





recover_flag = np.ones(r_cell.shape[1])
cell_type = np.ones(r_cell.shape[1]) * 1 # 1 for circle, 2 for ellipse, 3 for hypocycloid !!!!!!!!

### boundary points of each cell type
points_1 = dict()
for i in range(r_cell.shape[1]):
    points_1[i] = PolygonCoordinates_phi(poly_type, r_cell[:,i], R, poly_n, phi_cell[i])
   
h = distance(points_1[0][:,2],points_1[0][:,1]) 

area_int= PolyArea(points_1[0][0],points_1[0][1])
cell_area = np.zeros(r_cell.shape[1])
area_prev = np.zeros(r_cell.shape[1])
for i in range(r_cell.shape[1]):
    area_prev[i] = area_int

points_2 = dict()
for i in range(r_cell.shape[1]):
    points_2[i] = PolygonCoordinates_ell_phi(r_cell[:,i], ell_a, ell_b, poly_n, phi_cell[i])

points_3 = dict()
for i in range(r_cell.shape[1]):
    points_3[i] = PolygonCoordinates_hypo_phi(r_cell[:,i], hypo_a, hypo_b, poly_n, phi_cell[i])



### equilibrium vector of different shape of cells
cell_equi_vec_1=dict()
for i in range(r_cell.shape[1]):
    cell_equi_vec_1[i] = r_cell[:,i][:,newaxis]-points_1[i]

cell_equi_vec_2=dict()
for i in range(r_cell.shape[1]):
    cell_equi_vec_2[i] = r_cell[:,i][:,newaxis]-points_2[i]

cell_equi_vec_3 = dict()
for i in range(r_cell.shape[1]):
    cell_equi_vec_3[i] = r_cell[:, i][:, newaxis] - points_3[i]

#### to plot the points of each cell
if cell_type[0] == 1:
    points_plot = deepcopy(points_1)
elif cell_type[0]  == 2:
    points_plot = deepcopy(points_2)
else:
    points_plot = deepcopy(points_3)

for i in points_plot:
    points_plot[i] = np.append(points_plot[i], points_plot[i][:, 0][:, np.newaxis], axis=1)

############################################################
##FEniCS mesh settings
############################################################
domain_radius = 40
CompDomain=Circle(Point(0,0),domain_radius)
mesh = generate_mesh(CompDomain, 40)


meshpoints=mesh.coordinates()

parameters['reorder_dofs_serial'] = False




def repel_strength(d):
    return 0.5*max(sigma_rw-d,0)*exp(1/d)



def repel_force(point):
    result = np.array([0.0,0.0])
    result += (repel_strength(domain_radius-distance(point, np.array([0,0]))) * point * -1  / np.linalg.norm(point))
    #print(result[0],result[1])
    return result

lj_distance = 0.5
lj_strength = 0.01
def lennard_jones(d):
    d[d < lj_distance*0.6] = lj_distance*0.6
    return 4*lj_strength*((lj_distance/d)**12-(lj_distance/d)**6) 

def get_lj_forces(points):
    for i in range(0, r_cell.shape[1]-1):
        for j in range(i + 1, r_cell.shape[1]):
            dx = points[i][0][:, np.newaxis]-points[j][0]
            dy = points[i][1][:, np.newaxis]-points[j][1]
            distances = np.sqrt(dx**2 + dy**2)
            forces = lennard_jones(distances)
            fx = forces * dx
            fy = forces * dy

            points[i][0] += np.sum(fx, axis=1) * dt 
            points[i][1] += np.sum(fy, axis=1) * dt

            points[j][0] -= np.sum(fx, axis=0) * dt
            points[j][1] -= np.sum(fy, axis=0) * dt
            


### mesh info about index stuff
num_cells=mesh.num_cells()
cell_ver=dict() #cell is the key and vertex index+coordinate are the values
for cell in cells(mesh):
    cell_ver[cell.index()]=[]
    for vert in vertices(cell):
        cell_ver[cell.index()].append([vert.point().x(),vert.point().y()])
    for vert in vertices(cell):
        cell_ver[cell.index()].append(vert.index())

cell_midpoint=np.zeros([2,num_cells])
for cell in cells(mesh):
    cell_midpoint[:,cell.index()]=np.array([cell.midpoint().x(), cell.midpoint().y()])



def boundary(x, on_boundary):
    return on_boundary




### function space to solve weak form
V=FunctionSpace(mesh,'P',1)

v_1 = TensorElement('P', triangle, 1) # function element for eps (2*2)
v_2 = VectorElement('P', triangle, 1) # function element for v (2*1)
v = MixedElement([v_1, v_2])
V_morpho = FunctionSpace(mesh, v)

V_mechan=VectorFunctionSpace(mesh,'P',1)
dim_mechan=V_mechan.num_sub_spaces()

V_Tensor=TensorFunctionSpace(mesh,'P',1)

u_TGbeta_0 = Constant(0)
u_TGbeta_n=interpolate(u_TGbeta_0, V)
k_TGbeta=2.5 # magnitude of TGbeta
D_TGbeta=58.3 *4 #Diffusion (could be fct[Use Expression fct] or constant)

w_mechan_n=Function(V_morpho)

u_mechan_n_n = Function(V_mechan)
u_mechan_n = Function(V_mechan)
u_mechan = Function(V_mechan)
u_mechan_disp = Function(V_mechan)

u_mechan_vec = np.zeros((2, mesh.num_vertices()))
u_mechan_vec_norm = np.linalg.norm(u_mechan_vec, axis=0)
u_mechan_vec_norm_max = max(u_mechan_vec_norm)

bc_mechan=DirichletBC(V_morpho.sub(1), Constant((0,0)), boundary)


### Manually define Dirac delta
class Delta(Expression):
    def __init__(self, eps, x0, **kwargs):
        self.eps = eps
        self.x0 = x0
        # Expression.__init__(self, **kwargs)
    def eval(self, values, x):
        eps = self.eps
        values[0] = eps/pi/(np.linalg.norm(x-self.x0)**2 + eps**2)


kappa_TGbeta=100 #constant in Robin's BC
kappa_mechan=3 #constant in Robin's BC for mechanics


## mechanism preparation
def sigma_m(eps_mat, v):
    return Es / (1 + nu) * (eps_mat + nu / (1 - 2 * nu) * tr(eps_mat) * Identity(2)) + \
           10 * (Es/2/(1+nu) * sym(nabla_grad(v)) + Es/3/(1+nu)* tr(sym(nabla_grad(v))) * Identity(2))


def C_g(eps_mat):
    return 0.1*eps_mat


### check post process numerical results
M_mechan_max_list=[0.0]
u_TGbeta_max_list=[0.0]



###############################################################
#### preparation to calculate the orientation angle
###############################################################
def B_mat(phi):
    return np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])


##############################################################
##Loop preparation
#############################################################
## Cells Moving
n = 0 ### iteration

num_steps = 100
t = 0.0
T = dt*num_steps

vtkfile_TGbeta = File('u_TGbeta/solution.pvd')
vtkfile_cell = File('cell/solution.pvd')

while n<num_steps:
    print('*************************************', '\n', n, '\n')
    
    t_start = TM.time()

    r_cell_0 = np.array([]).reshape((2,-1)) # normal fibroblasts positions

    points_0 = dict() # points on the cell boundary
    for j in range(r_cell.shape[1]):
        points_0[j] = np.zeros((2, poly_n))

    
    # Define Jacobian for the moving mesh
    J_initial=Identity(dim_mechan)+nabla_grad(u_mechan_n)
    J_initial_inv=inv(J_initial)
    J_initial_det=sqrt(det(J_initial*J_initial.T))
    J_initial_det_fct=project(J_initial_det)

    J_initial_n = Identity(dim_mechan) + nabla_grad(u_mechan_n_n)
    J_initial_n_inv = inv(J_initial_n)
    J_initial_n_det = sqrt(det(J_initial_n * J_initial_n.T))
    J_initial_n_det_fct = project(J_initial_n_det)


    f_TGbeta = Function(V)
    ps_TGbeta = PointSource(V, Point(0.0, 0.0), 0.00005)
    ps_TGbeta.apply(f_TGbeta.vector())
    
    ### FEniCS for the point sources and immune cells situation
    ### Define the pointsource of concentration of TGbeta
    if max(u_TGbeta_n.vector().get_local()) == 0 or max(u_TGbeta_n.vector().get_local()) > 1E-7:
        u_TGbeta = TrialFunction(V)
        v = TestFunction(V)
        F_PDE = u_TGbeta * v * J_initial_det * dx - J_initial_n_det * u_TGbeta_n * v * dx + dt * kappa_TGbeta * u_TGbeta * v * J_initial_det * ds + \
                dt * D_TGbeta * inner(grad(u_TGbeta),
                                      grad(v)) * J_initial_det * dx - dt * f_TGbeta * v * J_initial_det * dx
        a_PDE, L_PDE = lhs(F_PDE), rhs(F_PDE)
        A_PDE, b_PDE = assemble_system(a_PDE, L_PDE)
        u_TGbeta = Function(V)
        

        solve(A_PDE, u_TGbeta.vector(), b_PDE)

        # extract gradu as a function
        gradu_TGbeta = project(grad(u_TGbeta))
        u_TGbeta_n.assign(u_TGbeta)
        vtkfile_TGbeta << (u_TGbeta_n, t)
        
        print('u_TGbeta_max: ', max(u_TGbeta.vector().get_local()))
        u_TGbeta_max_list.append(max(u_TGbeta.vector().get_local()))



    ### Define point forces
    f_mechan=Function(V_mechan)
    ps_0=[]
    ps_1=[]

   
    # r_immune_disp = np.array([]).reshape((2, -1))  # immune cells positions
    r_cell_disp = np.array([]).reshape((2, -1))  # normal fibroblasts positions
    #######################################################################
    ### skin cells/fibroblasts
    for i in range(r_cell.shape[1]):
        # calculate the angle of the orientation for cell i
        # determine displacement of the center and the boundary points
        c1 = r_cell[:, i]
        ### make sure the fibro are in the computational domain
        if not -domain_radius <= c1[0] <= domain_radius or not -domain_radius <= c1[1] <= domain_radius:
            continue

        if n==0:
            if cell_type[i] == 1:
                points = copy(points_1)
            elif cell_type[i] == 2:
                points = copy(points_2)
            elif cell_type[i] == 3:
                points = copy(points_3)

        if u_TGbeta_n(c1)>0.08:
            # cell_type[i] = 1
            # t_d = t

            cell_type[i] = 1
            t_d = t
            print('--------------------------------')
            print('%s -th cell differentiated' % i)
            print('--------------------------------')

           
        else:
            t_d=0

        if cell_type[i] == 1:
            points_ini = PolygonCoordinates_phi(poly_type, r_cell[:,0], R, poly_n, phi_cell[0])
            cell_equi_vec = copy(cell_equi_vec_1)
        elif cell_type[i] ==2:
            s = 1 - exp(-lam_s * (t - t_d))
            a_s = ell_a * s + R * (1 - s)
            b_s = ell_b * s + R * (1 - s)
            points_ini = PolygonCoordinates_ell_phi(r_cell[:, 0], a_s, b_s, poly_n, phi_cell[
                0])  # the position of points on the cell boundary if there is no deformation of cell
            cell_equi_vec = copy(cell_equi_vec_2)
        elif cell_type[i] == 3:
            s = 1 - exp(-lam_s * (t - t_d))
            a_s = hypo_a * s + R * (1 - s)
            b_s = hypo_b * s + R * (1 - s)
            points_ini = PolygonCoordinates_hypo_phi(r_cell[:, 0], a_s, b_s, poly_n, phi_cell[
                0])  # the position of points on the cell boundary if there is no deformation of cell
            cell_equi_vec = copy(cell_equi_vec_3)

        def obj_phi(phi):
            sum = 0
            for j in range(points[i].shape[1]):
                sum = sum + np.linalg.norm(np.dot(B_mat(pi), points_ini[:, j]) - points[i][:, j])

            return sum

        # cell_equi_vec = r_cell - points_ini
        theta = minimize_scalar(obj_phi, bounds=(0, 2 * pi)).x

        rw = sigma_rw * sqrt(dt) * np.random.multivariate_normal(mean, cov, 1).reshape((2,)) ## random walk


        for k in np.arange(np.shape(points[i])[1]):
            point1 = points[i][:, k]
            j = (k+1) % np.shape(points[i])[1]
            point2 = points[i][:, j]
            midpoint = (point1 + point2) / 2
            inwardn = inwardv(c1, point1, point2)
            if cell_type[i] == 3:
                ps_0 += [(Point(midpoint),
                        4 * P* dist(point1, point2) * inwardn[0])]
                ps_1 += [(Point(midpoint),
                      4 * P * dist(point1, point2) * inwardn[1])]
            else:
                ps_0 += [(Point(midpoint),
                          P * dist(point1, point2) * inwardn[0])]
                ps_1 += [
                    (Point(midpoint),
                     P * dist(point1, point2) * inwardn[1])]

        c = c1[:,newaxis]

        repel = np.array([0,0])
        


        for j in range(points[i].shape[1]):        
            # if (j == 0):
            #     tau[i][j] = (points[i][:,j+1] - points[i][:,j] * 2 + points[i][:,poly_n-1]) / \
            #             (distance(points[i][:,j+1],points[i][:,j])*distance(points[i][:,j],points[i][:,poly_n-1]))
                
            # elif (j == poly_n-1):
            #     tau[i][j] = (points[i][:,0] - points[i][:,j] * 2 + points[i][:,j-1]) / \
            #             (distance(points[i][:,0],points[i][:,j])*distance(points[i][:,j],points[i][:,j-1]))
            #     #print(tau[i][j][0],tau[i][j][1])
                
            # else:
            #     tau[i][j] = (points[i][:,j+1] - points[i][:,j] * 2 + points[i][:,j-1]) / \
            #             (distance(points[i][:,j+1],points[i][:,j])*distance(points[i][:,j],points[i][:,j-1]))
            
            force = repel_force(points[i][:, j])
            repel[0]  = force[0] if abs(force[0]) > abs(repel[0]) else repel[0]
            repel[1] = force[1] if abs(force[1]) > abs(repel[1]) else repel[1]

        lj_force = np.array([0.0,0.0])
        for m in range(r_cell.shape[1]):
            if m != i:
                lj_force += lennard_jones(distance(r_cell[:,m],r_cell[:,i]))*(r_cell[:,i]-r_cell[:,m])/np.linalg.norm(r_cell[:,m]-r_cell[:,i])



       
        for j in range(points[i].shape[1]):
            if max(u_TGbeta.vector().get_local())>1E-7 and all(np.linalg.norm(points[i],axis=0)>2) and recover_flag[i]:
                # points_0[i][:,j] = points[i][:, j] +\
                #     rw + \
                #     mu_TGbeta*gradu_TGbeta(points[i][:,j])/np.linalg.norm(gradu_TGbeta(c1))*dt +\
                #     np.array([lambda_curr[i],lambda_curr[i]]) * dt +\
                #           alpha*tau[i][j]*dt
                points_0[i][:,j] = points[i][:, j] + rw + alpha*np.array(tau[i][j])*dt +\
                      np.array([lambda_curr[i],lambda_curr[i]]) * dt + repel*dt
                #print(alpha*tau[j][0]*dt,alpha*tau[j][1]*dt)
                #points_0[i][:,j] = points[i][:, j] + alpha*np.array(tau[i][j])*dt
                

            else:
                recover_flag[i]=0
                points_0[i][:,j] = points[i][:, j] + rw + alpha*np.array(tau[i][j])*dt +\
                      np.array([lambda_curr[i],lambda_curr[i]]) * dt + repel * dt

       
                         
                       
            
        point = np.mean(points_0[i], axis=1)  ### shape (2,)
        r_cell_0 = np.append(r_cell_0, point[:, newaxis], axis=1)
        cell_centroid[i] = np.append(cell_centroid[i], point[:, newaxis], axis=1)
        



   

    get_lj_forces(points_0)
    # update the positions
    r_cell=deepcopy(r_cell_0)
    points=deepcopy(points_0)



    for key in points:
        fenics_mesh = create_boundary_mesh(points[key])
        # mesh_file = File(f"boundary_mesh_{key}_{step}.pvd")
        # mesh_file << fenics_mesh

        vtkfile_cell << (fenics_mesh , t)



    ## check the wound area by 3 dfferent approaches
    ## shoelace method to calculate the polygon area
    for j in range(r_cell.shape[1]):
        cell_area[j]= PolyArea(points[j][0],points[j][1])
        points_0[j] = np.append(points_0[j], points_0[j][:, 0][:, np.newaxis], axis=1)
        print('%s -th cell area(A_SL): %s' % (j, cell_area))
        print('%s -th cell area ratio(A_SL): %s' %(j, cell_area / (pi*R**2)))

    for j in range(r_cell.shape[1]):
        dlambda = (beta_one*lambda_curr[i]*(cell_area[j]-area_int+(cell_area[j]-area_prev[j])/dt))/(area_int*(lambda_curr[i]+beta_one))-\
            beta_two*lambda_curr[i]

        lambda_curr[i] = dt*dlambda+lambda_curr[i]
        area_prev[j] = cell_area[j]
        print("next lambda value:", lambda_curr[i])
    
    ### update cytokines concentration
    u_TGbeta_n.assign(u_TGbeta)

    ### iteration update
    n = n + 1
    ### time update
    t = t + dt
    t_end = TM.time()
    print("------ %s seconds ------" % (t_end - t_start))

    gc.collect()

    ### live plot
    

    plt.clf()
    plt.ion()
    plt.axis('equal')
    # # plt.figure('Simulations')
    plt.xlim([-domain_radius, domain_radius])
    plt.ylim([-domain_radius, domain_radius])
    plt.tick_params(which='major', labelsize=30)
    plt.scatter(0, 0, color='red')
    # plt.title(n)
    
    # dolfin.plot(u_TGbeta)
    #dolfin.plot(mesh, title = 'mesh')
   
    for j in np.arange(r_cell.shape[1]):
        #plt.plot(r_cell[0, j] + R * np.cos(plot_phi), r_cell[1, j] + R * np.sin(plot_phi), linewidth=0.9,color='blue')
       
        plt.plot(points_0[j][0], points_0[j][1],color='blue',linewidth=2.5)


        #plt.plot(points_plot[j][0], points_plot[j][1], color='red',linewidth=2.5)

        # for i in range(points[i].shape[1]):
        #     xvec = np.array([points_plot[j][:,i][0], r_cell_plot[0][j]])
        #     yvec = np.array([points_plot[j][:,i][1], r_cell_plot[1][j]])
        #     xvec_new = np.array([points[j][:,i][0], r_cell[0][j]])
        #     yvec_new = np.array([points[j][:,i][1], r_cell[1][j]])
            #plt.plot(xvec, yvec, color='black',linewidth=0.5)
            #plt.plot(xvec_new,yvec_new,color = 'black',linewidth=0.5)

        x_coords = points[j][0]
        y_coords = points[j][1]

        


        # Fill the region enclosed by the points with white color
        plt.fill(x_coords, y_coords, 'white')
        plt.scatter(cell_centroid[j][0],cell_centroid[j][1],s=0.1,color='black')
        plt.plot(cell_centroid[j][0], cell_centroid[j][1],markersize=0.3, color='black')

    
   



   

    plt.pause(0.0001)
    plt.show()




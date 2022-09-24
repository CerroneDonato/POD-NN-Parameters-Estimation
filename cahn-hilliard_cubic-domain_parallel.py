import os

import numpy as np
import sys
import ufl
from dolfinx.common import IndexMap
from dolfinx import log, plot ,fem
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_cube, locate_entities
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, variable, as_ufl, div
from math import sqrt
from mpi4py import MPI
from petsc4py import PETSc
from ufl.operators import max_value

np.set_printoptions(threshold=sys.maxsize)
log.set_output_file("log.txt")


epsilon = 0.079 # surface parameter
dt = 1          # time step
theta = 0.5     # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
M0= 390000      #tumor inter-phase friction
nu=1            #tumor cells proliferation rate
tresh=0.15      #hypoxia threshold
k=622.7         #young modulus
Dn=155.52       #oxygen diffusion coefficient
deltan=15552    #oxygen consumption rate
Sn=10000        #oxygen supply rate
s=5.07          #steepness coefficient of fractional tumor cell kill
lamb=0.9        #exponent of fractional tumor cell kill
d=2.8           #saturation level of fractional tumor cell kill

R_eff=0.0648      #Radiotherapy death rate
k_C1=0.235        #Concomitant chemotherapy death rate
k_C2=0.747        #First cycle of adjuvant CHT death rate
k_C3=0.796        #Remaining cycles of adjuvant CHT death rate

#Chemoterapic cycles
s0=0
s1=20
s2=25
s3=50

T = 50 * dt     #Total time

comm = MPI.COMM_WORLD

msh = create_unit_cube(MPI.COMM_WORLD, 15, 15, 15, CellType.tetrahedron) #Cration of the mesh
P1 = ufl.VectorElement("Lagrange", msh.ufl_cell(), 1,3)
ME = fem.FunctionSpace(msh,P1)                                           #Cration of the FUNCTION SPACE

#Definition of cell death rate 
k_C = Constant(msh, PETSc.ScalarType(0.0))
k_R = Constant(msh, PETSc.ScalarType(R_eff))

u = Function(ME)   #solution step n+1
phi, mu, n =ufl.split(u) # splitting the solution in the three component:
                         # phi = tumor volume fraction - liquid phase volume fraction
                         # mu = chemical potential
                         # n = concentration of oxygen

u0 = Function(ME)  #solution step n
phi0, mu0, n0=ufl.split(u0)

vv = ufl.TestFunction(ME) #test function

u.x.array[:] = 0.0
u0.x.array[:] = 0.0

#INITIAL CONDITION
u.sub(0).interpolate(lambda x : -1+2*(((x[0]-0.5)**2+(x[1]-0.5)**2+(x[2]-0.5)**2)<0.01))
u.x.scatter_forward()
u0.x.array[:] = u.x.array

#Definition of the output file
file = XDMFFile(MPI.COMM_WORLD, "ch_cubic_parallel/out_tumor.xdmf", "w")
filen = XDMFFile(MPI.COMM_WORLD, "ch_cubic_parallel/out_liquidphase.xdmf", "w")
#Plotting the mesh
file.write_mesh(msh)
filen.write_mesh(msh)

#VARATIONAL FORMULATION
##Definition of the potential
phi = ufl.variable(phi)
f = (phi**4 + 1)/4
dfdphi = ufl.diff(f, phi)

phi0 = ufl.variable(phi0)
f0 = -0.5*(phi0**2)
df0dphi0 = ufl.diff(f0, phi0)

#Definition of auxiliary functions
h=0.5*(1+phi0)
g=(2-phi0)/3

mu_mid = (1.0 - theta) * mu0 + theta * mu

###FORMULATION###
F0 = inner(phi, vv[0]) * dx - inner(phi0, vv[0]) * dx + dt * (1/M0) * inner(grad(mu_mid), grad(vv[0])) * dx - nu * dt * inner((n-tresh)*h,vv[0]) * dx + dt * inner((k_R+k_C)*h,vv[0]) * dx
F1 = inner(mu, vv[1]) * dx - k * inner(dfdphi, vv[1]) * dx - k * inner(df0dphi0, vv[1]) * dx - (epsilon**2) * inner(grad(phi), grad(vv[1])) * dx
F2 = inner(n, vv[2]) * dx - inner(n0, vv[2]) * dx + dt * Dn * inner(grad(n),grad(vv[2])) * dx - dt * Sn * inner((1-n)*g,vv[2]) * dx + deltan * dt *inner(n*h,vv[2])*dx

F = F0 + F1 +F2

problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

phi = u.sub(0)
n = u.sub(2)

t = 0.0

##RESOLUTION##
while (t < T):
#therapeutic schedule (fictional)
    if (t >= s0) & (t<=s1):
        k_C.value=PETSc.ScalarType(k_C1)
    elif (t >= s2) & (t<=s3):
        k_C.value=PETSc.ScalarType(k_C2)
    else :
        k_C.value=PETSc.ScalarType(k_C3)

    r = solver.solve(u) # actual computaion of the variational formulation

    if comm.rank==0:
        print(f"Step {int(t/dt)}: num iterations: {r[0]}")
    u0.x.array[:] = u.x.array
    file.write_function(phi, t)
    filen.write_function(n, t)
    t += dt

file.close()
filen.close()

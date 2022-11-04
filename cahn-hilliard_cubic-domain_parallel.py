import os
import gmsh
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
from ufl.operators import max_value,min_value
from xml.dom import minidom

np.set_printoptions(threshold=sys.maxsize)
#log.set_output_file("log.txt")

file_in = minidom.parse('11-09-18/d11S.xml')
tensor_values = file.getElementsByTagName('value')

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "11-09-18/meshR.xdmf", "r") as xdmf:
    msh=xdmf.read_mesh(name="Grid")

theta=0.5
dt=3600
nu = Constant(msh, PETSc.ScalarType(0.3/86400))                   # tumor cells proliferation rate  (1/s)
M0 = Constant(msh, PETSc.ScalarType(3000*86.4))                  # tumor inter-phase friction (kg/(s*mm^3))
epsilon = Constant(msh, PETSc.ScalarType(0.9*pow(10,-3/2)))      # diffuse interphase thickness ((kg*mm/s^2)^(1/2))
k = Constant(msh, PETSc.ScalarType(694*pow(10,-3)))           # Young modulus (kg/(mm*s^2))
tresh = Constant(msh, PETSc.ScalarType(0.325))                    # hypoxia threshold (adim)
Dn = Constant(msh, PETSc.ScalarType(86.4/86400))                 # oxygen diffusion coefficient (mm^2/s)
deltan = Constant(msh, PETSc.ScalarType(8640/86400))             # oxygen consumption rate (1/s)
Sn = Constant(msh, PETSc.ScalarType(10000/86400))

R_eff=0.0      #Radiotherapy death rate
k_C1=0.0        #Concomitant chemotherapy death rate
k_C2=0.6        #First cycle of adjuvant CHT death rate
k_C3=0.0        #Remaining cycles of adjuvant CHT death rate

#Chemoterapic cycles
s0=0
s1=20
s2=25
s3=50

T = 2000 * dt     #Total time

comm = MPI.COMM_WORLD

#msh = create_unit_cube(MPI.COMM_WORLD, 50, 50, 50, CellType.tetrahedron) #Cration of the mesh
P1 = ufl.VectorElement("Lagrange", msh.ufl_cell(), 1,3)
ME = fem.FunctionSpace(msh,P1)                                           #Cration of the FUNCTION SPACE

P_DG = ufl.VectorElement("DG", msh.ufl_cell(), 0,1)
M_const = fem.FunctionSpace(msh,P1)

Pphi, Pphi_to_P1 = ME.sub(0).collapse()
Pmu, Pmu_to_P1 = ME.sub(1).collapse()                                        #Cration of the FUNCTION SPACE
Pn, Pn_to_P1 = ME.sub(2).collapse()
#Definition of cell death rate
k_C = Constant(msh, PETSc.ScalarType(0.0))
k_R = Constant(msh, PETSc.ScalarType(R_eff))

phi_init=Function(Pphi)
mu_init=Function(Pmu)
n_init=Function(Pn)

u = Function(ME)   #solution step n+1
phi, mu, n =ufl.split(u) # splitting the solution in the three component:
                         # phi = tumor volume fraction - liquid phase volume fraction
                         # mu = chemical potential
                         # n = concentration of oxygen

u0 = Function(ME)  #solution step n
phi0, mu0, n0=ufl.split(u0)

v_phi=ufl.TestFunction(Pphi)
v_mu=ufl.TestFunction(Pmu)
v_n=ufl.TestFunction(Pn)

vv = ufl.TestFunction(ME) #test function

#Definition of the output file
file = XDMFFile(MPI.COMM_WORLD, "ch_cubic_para/out_tumor.xdmf", "w")
filen = XDMFFile(MPI.COMM_WORLD, "ch_cubic_para/out_liquidphase.xdmf", "w")
#Plotting the mesh
file.write_mesh(msh)
filen.write_mesh(msh)
t = 0.0

u.x.array[:] = 0.0
u0.x.array[:] = 0.0

#INITIAL CONDITION
phi_init.interpolate(lambda x : -1+1.8*(((x[0]-0.5)**2+(x[1]-0.5)**2+(x[2]-0.5)**2)<0.01))
#phi_init.interpolate(lambda x : -1+1.8*np.exp(((x[0]-0.5)**2+(x[1]-0.5)**2+(x[2]-0.5)**2)*-10))
phi_init.x.scatter_forward()
u.x.array[Pphi_to_P1]=phi_init.x.array
u.x.scatter_forward()
#Definition of auxiliary functions
Fmu = inner(mu_init, v_mu) * dx - k * inner(phi_init**3, v_mu) * dx - k * inner(-phi_init, v_mu) * dx - (epsilon**2) * inner(grad(phi_init), grad(v_mu)) * dx

problem = NonlinearProblem(Fmu, mu_init)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

r = solver.solve(mu_init) # actual computaion of the variational formulation
mu_init.x.scatter_forward()
u.x.array[Pmu_to_P1]=mu_init.x.array
u.x.scatter_forward()
h=0.5*(1+phi_init)
g=(2-phi_init)/3

Fn =  Dn * inner(grad(n_init),grad(v_n)) * dx - Sn * inner((1-n_init)*g,v_n) * dx + deltan *inner(n_init*h,v_n)*dx
problem = NonlinearProblem(Fn, n_init)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

r = solver.solve(n_init) # actual computaion of the variational formulation
n_init.x.scatter_forward()
u.x.array[Pn_to_P1]=n_init.x.array
u.x.scatter_forward()
u0.x.array[:] = u.x.array
u0.x.scatter_forward()




#f = (phi**4 + 1)/4
#dfdphi = ufl.diff(f, phi)

dfdphi=phi**3

#phi0 = ufl.variable(phi0)
#f0 = -0.5*(phi0**2)
#df0dphi0 = ufl.diff(f0, phi0)
df0dphi0 = -phi0

#Definition of auxiliary functions
#h=0.5*(1+phi0)
h=max_value(min_value(1,0.5*(1+phi0)),0)
#g=(2-phi0)/3
g=max_value(min_value(1,0.5*(2-phi0)/3),1/3)

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
file.write_function(phi, t)
filen.write_function(n, t)
##RESOLUTION##
while (t < T):
    t += dt
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


file.close()
filen.close()

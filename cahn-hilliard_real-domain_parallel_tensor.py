import gmsh
import dolfinx.io
from dolfinx.io import XDMFFile
from mpi4py import MPI
import dolfinx.mesh
from dolfinx.cpp.mesh import CellType
import os
import numpy as np
import sys
import ufl
from dolfinx.common import IndexMap
from dolfinx import log, plot ,fem
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_cube, locate_entities, create_mesh
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, variable, as_ufl, div
from math import sqrt
from petsc4py import PETSc
from ufl.operators import max_value,min_value
from ufl.tensors import as_tensor, as_matrix

np.set_printoptions(threshold=sys.maxsize)
#D_array=np.load("tensor_data/diffusion_array.npy")
comm=MPI.COMM_WORLD

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensord11.xdmf", "r") as xdmf:
    msh=xdmf.read_mesh(name="Grid")
    tag11=xdmf.read_meshtags(msh,"Grid")

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensord12.xdmf", "r") as xdmf:
    #msh1=xdmf.read_mesh(name="Grid")
    tag12=xdmf.read_meshtags(msh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensord13.xdmf", "r") as xdmf:
    #msh1=xdmf.read_mesh(name="Grid")
    tag13=xdmf.read_meshtags(msh,"Grid")

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensord22.xdmf", "r") as xdmf:
    #msh1=xdmf.read_mesh(name="Grid")
    tag22=xdmf.read_meshtags(msh,"Grid")

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensord23.xdmf", "r") as xdmf:
    #msh1=xdmf.read_mesh(name="Grid")
    tag23=xdmf.read_meshtags(msh,"Grid")

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensord33.xdmf", "r") as xdmf:
    #msh1=xdmf.read_mesh(name="Grid")
    tag33=xdmf.read_meshtags(msh,"Grid")

valuesD11=tag11.values/1000000
valuesD12=tag12.values/1000000
valuesD13=tag13.values/1000000
valuesD22=tag22.values/1000000
valuesD23=tag23.values/1000000
valuesD33=tag33.values/1000000
element = ufl.FiniteElement("DG", msh.ufl_cell(), 0)
PD = ufl.MixedElement(element,element,element,element,element,element)
MD = fem.FunctionSpace(msh,PD)

Pd11, Pd11_to_P1 = MD.sub(0).collapse()
Pd12, Pd12_to_P1 = MD.sub(1).collapse()
Pd13, Pd13_to_P1 = MD.sub(2).collapse()
Pd22, Pd22_to_P1 = MD.sub(3).collapse()
Pd23, Pd23_to_P1 = MD.sub(4).collapse()
Pd22, Pd33_to_P1 = MD.sub(5).collapse()

D = Function(MD)
D11,D12,D13,D22,D23,D33=D.split()

D.x.array[Pd11_to_P1]=valuesD11
D.x.array[Pd12_to_P1]=valuesD12
D.x.array[Pd13_to_P1]=valuesD13
D.x.array[Pd22_to_P1]=valuesD22
D.x.array[Pd23_to_P1]=valuesD23
D.x.array[Pd33_to_P1]=valuesD33

fd = XDMFFile(MPI.COMM_WORLD, "out/ten_D11.xdmf", "w")
fd.write_mesh(msh)
fd.write_function(D11)
fd = XDMFFile(MPI.COMM_WORLD, "out/ten_D12.xdmf", "w")
fd.write_mesh(msh)
fd.write_function(D12)
fd = XDMFFile(MPI.COMM_WORLD, "out/ten_D13.xdmf", "w")
fd.write_mesh(msh)
fd.write_function(D13)
fd = XDMFFile(MPI.COMM_WORLD, "out/ten_D22.xdmf", "w")
fd.write_mesh(msh)
fd.write_function(D22)
fd = XDMFFile(MPI.COMM_WORLD, "out/ten_D23.xdmf", "w")
fd.write_mesh(msh)
fd.write_function(D23)
fd = XDMFFile(MPI.COMM_WORLD, "out/ten_D33.xdmf", "w")
fd.write_mesh(msh)
fd.write_function(D33)

#with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensord11.xdmf", "r") as xdmf:
#    msh=xdmf.read_mesh(name="Grid")
#    tag=xdmf.read_meshtags(msh,"Grid")
#print("Process ",comm.rank," : ",tag.values[0])
#valuesD11=tag.values/1000000
#PD11 = ufl.VectorElement("DG", msh.ufl_cell(), 0, 1)
#MD11 = fem.FunctionSpace(msh,PD11)
#D11 = Function(MD11)
#D11.x.array[:]=valuesD11
#fd = XDMFFile(MPI.COMM_WORLD, "out/ten_D11.xdmf", "w")
#fd.write_mesh(msh)
#fd.write_function(D11)



file = XDMFFile(MPI.COMM_WORLD, "output/out_tumor.xdmf", "w")
filen = XDMFFile(MPI.COMM_WORLD, "output/out_nutrient.xdmf", "w")
#Plotting the mesh
file.write_mesh(msh)
filen.write_mesh(msh)


theta=0.5
dt=300
nu = Constant(msh, PETSc.ScalarType(1/86400))                   # tumor cells proliferation rate  (1/s)
M0 = Constant(msh, PETSc.ScalarType(1000*86.4))                  # tumor inter-phase friction (kg/(s*mm^3))
epsilon = Constant(msh, PETSc.ScalarType(0.25))      # diffuse interphase thickness ((kg*mm/s^2)^(1/2))
k = Constant(msh, PETSc.ScalarType(394*pow(10,-3)))           # Young modulus (kg/(mm*s^2))
tresh = Constant(msh, PETSc.ScalarType(0.125))                    # hypoxia threshold (adim)
Dn = Constant(msh, PETSc.ScalarType(86.4/86400))                 # oxygen diffusion coefficient (mm^2/s)
deltan = Constant(msh, PETSc.ScalarType(8640/86400))             # oxygen consumption rate (1/s)
Sn = Constant(msh, PETSc.ScalarType(10000/86400))          #saturation level of fractional tumor cell kill

R_eff=0.0      #Radiotherapy death rate
k_C1=0.0        #Concomitant chemotherapy death rate
k_C2=5/86400       #First cycle of adjuvant CHT death rate
k_C3=0.0        #Remaining cycles of adjuvant CHT death rate

#Chemoterapic cycles
s0=0
s1=20
s2=50*dt
s3=100*dt

T = 0 * dt    #Total time

comm = MPI.COMM_WORLD
a=0.001


#PD = ufl.VectorElement("DG", msh.ufl_cell(), 0, 6)
#MD = fem.FunctionSpace(msh,PD)

#Pd11, Pd11_to_P1 = MD.sub(0).collapse()
#Pd12, Pd12_to_P1 = MD.sub(1).collapse()
#Pd13, Pd13_to_P1 = MD.sub(2).collapse()
#Pd22, Pd22_to_P1 = MD.sub(3).collapse()
#Pd23, Pd23_to_P1 = MD.sub(4).collapse()
#Pd22, Pd33_to_P1 = MD.sub(5).collapse()

#D = Function(MD)
#D11,D12,D13,D22,D23,D33=D.split()



print("Interpolated")


conversion_factor = Constant(msh, PETSc.ScalarType(1/86400))
matrixD=as_matrix([[D11*conversion_factor,D12*conversion_factor,D13*conversion_factor],[D12*conversion_factor,D22*conversion_factor,D23*conversion_factor],[D13*conversion_factor,D23*conversion_factor,D33*conversion_factor]])


element1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
P1 = ufl.MixedElement(element1,element1,element1)
#P1 = ufl.VectorElement("Lagrange", msh.ufl_cell(), 1,3)
ME = fem.FunctionSpace(msh,P1)                                           #Cration of the FUNCTION SPACE


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


t = 0.0

u.x.array[:] = 0.0
u0.x.array[:] = 0.0


phi_init.interpolate(lambda x : 1.998*np.exp(-0.01*((x[0]-193.3975)**2+(x[1]-308)**2+(x[2]-12)**2))-0.999)

phi_init.x.scatter_forward()

u.x.array[Pphi_to_P1]=phi_init.x.array
u.x.scatter_forward()

phi_init = ufl.variable(phi)
f_init= (phi_init**4 + 1)/4-a*ufl.ln(1-phi_init)-a*ufl.ln(1+phi_init)
#f = (phi**4 + 1)/4
dfdphi_init = ufl.diff(f_init, phi_init)
#Definition of auxiliary functions
Fmu = inner(mu_init, v_mu) * dx - k * inner(dfdphi_init, v_mu) * dx - k * inner(-phi_init, v_mu) * dx - (epsilon**2) * inner(grad(phi_init), grad(v_mu)) * dx

problem = NonlinearProblem(Fmu, mu_init)
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
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

r = solver.solve(n_init) # actual computaion of the variational formulation
n_init.x.scatter_forward()
u.x.array[Pn_to_P1]=n_init.x.array
u.x.scatter_forward()
u0.x.array[:] = u.x.array
u0.x.scatter_forward()



if comm.rank==0:
    print(f"Initial condition computed")


phi = ufl.variable(phi)
f = (phi**4 + 1)/4-a*ufl.ln(1-phi)-a*ufl.ln(1+phi)
#f = (phi**4 + 1)/4
dfdphi = ufl.diff(f, phi)

#dfdphi=phi**3

phi0 = ufl.variable(phi0)
f0 = -0.5*(phi0**2)
df0dphi0 = ufl.diff(f0, phi0)
#df0dphi0 = -phi0

#Definition of auxiliary functions
#h=0.5*(1+phi0)
h=max_value(min_value(1,0.5*(1+phi0)),0)
#g=(2-phi0)/3
g=max_value(min_value(1,0.5*(2-phi0)/3),1/3)

mu_mid = (1.0 - theta) * mu0 + theta * mu

###FORMULATION###
F0 = inner(phi, vv[0]) * dx - inner(phi0, vv[0]) * dx + dt * (1/M0) * inner(matrixD*grad(mu_mid), grad(vv[0])) * dx - nu * dt * inner((n-tresh)*h,vv[0]) * dx + dt * inner((k_R+k_C)*h,vv[0]) * dx
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

import gmsh
import dolfinx.io
from dolfinx.io import XDMFFile, extract_gmsh_topology_and_markers, distribute_entity_data
from mpi4py import MPI
import dolfinx.mesh
import os
#Prova push
import numpy as np
import sys
import ufl
from dolfinx.common import IndexMap
from dolfinx import log, plot ,fem
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, variable, as_ufl, div
from math import sqrt
from mpi4py import MPI
from petsc4py import PETSc
from ufl.operators import max_value,min_value
from ufl.tensors import as_tensor, as_matrix
from xml.dom import minidom
import meshio

# read the 3D GMsh file:
np.set_printoptions(threshold=sys.maxsize)

msh = meshio.read('brain.xml')
type="tetra"
x=msh.points
topology=msh.cells

file11 = minidom.parse('tensor_data/d11S.xml')
d11 = file11.getElementsByTagName('value')
print("Parsing finished")
file12 = minidom.parse('tensor_data/d12S.xml')
d12 = file12.getElementsByTagName('value')
print("Parsing finished")
file13 = minidom.parse('tensor_data/d13S.xml')
d13 = file13.getElementsByTagName('value')
print("Parsing finished")
file22 = minidom.parse('tensor_data/d22S.xml')
d22 = file22.getElementsByTagName('value')
print("Parsing finished")
file23 = minidom.parse('tensor_data/d23S.xml')
d23 = file23.getElementsByTagName('value')
print("Parsing finished")
file33 = minidom.parse('tensor_data/d33S.xml')
d33 = file33.getElementsByTagName('value')
print("Parsing finished")


data11=np.zeros(196778,dtype=int)
data12=np.zeros(196778,dtype=int)
data13=np.zeros(196778,dtype=int)
data22=np.zeros(196778,dtype=int)
data23=np.zeros(196778,dtype=int)
data33=np.zeros(196778,dtype=int)
#data=np.zeros(196778)
for ii in range(len(data11)):
 #data[ii]=int(1000000000*float(d11[ii].attributes['value'].value))
 data11[ii]=1000000*float(d11[ii].attributes['value'].value)
 data12[ii]=1000000*float(d12[ii].attributes['value'].value)
 data13[ii]=1000000*float(d13[ii].attributes['value'].value)
 data22[ii]=1000000*float(d22[ii].attributes['value'].value)
 data23[ii]=1000000*float(d23[ii].attributes['value'].value)
 data33[ii]=1000000*float(d33[ii].attributes['value'].value)

mesh11 = meshio.Mesh(
    x,
    topology,
    cell_data={"tetra":[data11]}
)
mesh12 = meshio.Mesh(
    x,
    topology,
    cell_data={"tetra":[data12]}
)
mesh13 = meshio.Mesh(
    x,
    topology,
    cell_data={"tetra":[data13]}
)
mesh22 = meshio.Mesh(
    x,
    topology,
    cell_data={"tetra":[data22]}
)
mesh23 = meshio.Mesh(
    x,
    topology,
    cell_data={"tetra":[data23]}
)
mesh33 = meshio.Mesh(
    x,
    topology,
    cell_data={"tetra":[data33]}
)

meshio.write("tensord11.xdmf", mesh11)
meshio.write("tensord12.xdmf", mesh12)
meshio.write("tensord13.xdmf", mesh13)
meshio.write("tensord22.xdmf", mesh22)
meshio.write("tensord23.xdmf", mesh23)
meshio.write("tensord33.xdmf", mesh33)
print("Done")
#fd = XDMFFile(MPI.COMM_WORLD, "create.xdmf", "w")
#fd.write_mesh(mshnew)


#meshio.write("mesh.xdmf", meshio.Mesh(points=msh.points,cells={"tetra": msh.cells["tetra"]}))
#with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
#    msh=xdmf.read_mesh(name="Grid")
#    tag11=xdmf.read_meshtags(msh,"Grid")





#MD = fem.FunctionSpace(msh,("DG", 0))
#u = Function(MD)

#u.x.array[:]=tag11.values

#file = XDMFFile(MPI.COMM_WORLD, "out_nutrient.xdmf", "w")
#Plotting the mesh
#file.write_mesh(msh)
#file.write_function(u)
#print(u.x.array)

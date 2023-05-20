import gmsh
import dolfinx.io
from dolfinx.io import XDMFFile
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

msh = meshio.read('tensor_data/brain.xml')
type="tetra"
x=msh.points #LUNGHEZZA CARATTERISTICA
topology=msh.cells


mesh11 = meshio.Mesh(
    x,
    topology
)


meshio.write("tensord11.xdmf", mesh11)

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

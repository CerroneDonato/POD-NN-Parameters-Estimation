import typing
import itertools
import dolfinx.fem
import dolfinx.mesh


import mpi4py.MPI
import multiphenicsx.fem
import multiphenicsx.io
import numpy as np
import petsc4py.PETSc
import ufl
import abc
from mpi4py import MPI
import sys
import os
from pathlib import Path
import rbnicsx.backends
import rbnicsx.online
import rbnicsx.test
import plotly.graph_objects as go
from ufl.tensors import as_matrix
import time

tic=time.time()

###READ MESH AND TENSORS ###

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensord11.xdmf", "r") as xdmf:
    mesh=xdmf.read_mesh(name="Grid")
    tag11=xdmf.read_meshtags(mesh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensord12.xdmf", "r") as xdmf:
    tag12=xdmf.read_meshtags(mesh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensord13.xdmf", "r") as xdmf:
    tag13=xdmf.read_meshtags(mesh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensord22.xdmf", "r") as xdmf:
    tag22=xdmf.read_meshtags(mesh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensord23.xdmf", "r") as xdmf:
    tag23=xdmf.read_meshtags(mesh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensord33.xdmf", "r") as xdmf:
    tag33=xdmf.read_meshtags(mesh,"Grid")

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensort11.xdmf", "r") as xdmf:
    tag11t=xdmf.read_meshtags(mesh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensort12.xdmf", "r") as xdmf:
    tag12t=xdmf.read_meshtags(mesh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensort13.xdmf", "r") as xdmf:
    tag13t=xdmf.read_meshtags(mesh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensort22.xdmf", "r") as xdmf:
    tag22t=xdmf.read_meshtags(mesh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensort23.xdmf", "r") as xdmf:
    tag23t=xdmf.read_meshtags(mesh,"Grid")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tensori/tensort33.xdmf", "r") as xdmf:
    tag33t=xdmf.read_meshtags(mesh,"Grid")

valuesD11=tag11.values/1000000
valuesD12=tag12.values/1000000
valuesD13=tag13.values/1000000
valuesD22=tag22.values/1000000
valuesD23=tag23.values/1000000
valuesD33=tag33.values/1000000
valuesT11=tag11t.values/1000000
valuesT12=tag12t.values/1000000
valuesT13=tag13t.values/1000000
valuesT22=tag22t.values/1000000
valuesT23=tag23t.values/1000000
valuesT33=tag33t.values/1000000

element = ufl.FiniteElement("DG", mesh.ufl_cell(), 0)
PD = ufl.MixedElement(element,element,element,element,element,element)
MD = dolfinx.fem.FunctionSpace(mesh,PD)
Pd11, Pd11_to_P1 = MD.sub(0).collapse()
Pd12, Pd12_to_P1 = MD.sub(1).collapse()
Pd13, Pd13_to_P1 = MD.sub(2).collapse()
Pd22, Pd22_to_P1 = MD.sub(3).collapse()
Pd23, Pd23_to_P1 = MD.sub(4).collapse()
Pd22, Pd33_to_P1 = MD.sub(5).collapse()
D = dolfinx.fem.Function(MD)
D11,D12,D13,D22,D23,D33=D.split()
T = dolfinx.fem.Function(MD)
T11,T12,T13,T22,T23,T33=T.split()

D.x.array[Pd11_to_P1]=valuesD11
D.x.array[Pd12_to_P1]=valuesD12
D.x.array[Pd13_to_P1]=valuesD13
D.x.array[Pd22_to_P1]=valuesD22
D.x.array[Pd23_to_P1]=valuesD23
D.x.array[Pd33_to_P1]=valuesD33

T.x.array[Pd11_to_P1]=valuesT11
T.x.array[Pd12_to_P1]=valuesT12
T.x.array[Pd13_to_P1]=valuesT13
T.x.array[Pd22_to_P1]=valuesT22
T.x.array[Pd23_to_P1]=valuesT23
T.x.array[Pd33_to_P1]=valuesT33

fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_D11.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(D11)
fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_D12.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(D12)
fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_D13.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(D13)
fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_D22.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(D22)
fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_D23.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(D23)
fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_D33.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(D33)

fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_T11.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(T11)
fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_T12.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(T12)
fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_T13.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(T13)
fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_T22.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(T22)
fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_T23.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(T23)
fd = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/ten_T33.xdmf", "w")
fd.write_mesh(mesh)
fd.write_function(T33)


if MPI.COMM_WORLD.size > 1:
    serial_or_parallel = "parallel"
else:
    serial_or_parallel = "serial"



### CLASS DEFINITION ###

class Problem(object):
    """Define a nonlinear problem, and solve it with SNES."""

    def __init__(self) -> None:
        # Define function space
        element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        V_phi = dolfinx.fem.FunctionSpace(mesh, element)
        V_mu = dolfinx.fem.FunctionSpace(mesh, element)
        V_n = dolfinx.fem.FunctionSpace(mesh, element)

        self._V = (V_phi,V_mu,V_n)
        self._function_space_phi=V_phi
        self._function_space_mu=V_mu
        self._function_space_n=V_n
        # Define trial and test functions
        (v_phi,v_mu,v_n) = (ufl.TestFunction(V_phi), ufl.TestFunction(V_mu),ufl.TestFunction(V_n))
        (dphi, dmu, dn) = (ufl.TrialFunction(V_phi), ufl.TrialFunction(V_mu),ufl.TrialFunction(V_n))
        # Define solution components
        (phi,mu,n) = (dolfinx.fem.Function(self._V[0]), dolfinx.fem.Function(self._V[1]),dolfinx.fem.Function(self._V[2]))
        (phi0,mu0,n0) = (dolfinx.fem.Function(self._V[0]), dolfinx.fem.Function(self._V[1]),dolfinx.fem.Function(self._V[2]))

        self._trial_dphi = dphi
        self._test_v_phi   = v_phi
        self._trial_dn = dn
        self._test_v_n   = v_n
        self._trial_dmu = dmu
        self._test_v_mu   = v_mu
        self._solutions = (phi,mu,n)
        self._solutions0 = (phi0,mu0,n0)

        L_car=10.0
        T_car=1.0

        #param=[1.0, 467.3884497716478, 0.13379603788437794, 1.0, 0.3306644170256226, 1090218.193428378, 0.20187865621588333, 5106169.530219245]
        e_symb = rbnicsx.backends.SymbolicParameters(mesh, shape=(6, ))
        self._e_symb = e_symb

        epsilon_car=30 #e_symb[7]

        nu = e_symb[5]*T_car #param[0]
        M0 = e_symb[1]*(L_car**4)/((epsilon_car**2)*T_car) #param[1]
        epsilon =1.0     #param[2]
        k = e_symb[2]*(L_car**2)/(epsilon_car**2) #param[3]
        tresh = e_symb[3]   #dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(param[4]))
        deltan = e_symb[4]*T_car #param[5]
        Dn = deltan*T_car/(L_car**2)  #param[6]
        Sn = e_symb[0]*T_car #param[7]
        theta=0.5
        # Time Discretization
        dt = 0.5*T_car
        self._dt = dt
        T  = 60 * dt
        self._T  = T
        conversion_factor = 1.0
        matrixD=as_matrix([[D11*conversion_factor,D12*conversion_factor,D13*conversion_factor],[D12*conversion_factor,D22*conversion_factor,D23*conversion_factor],[D13*conversion_factor,D23*conversion_factor,D33*conversion_factor]])
        matrixT=as_matrix([[T11,T12,T13],[T12,T22,T23],[T13,T23,T33]])
        mu_mid = (1.0 - theta) * mu0 + theta * mu
        # Define residual block form of the problem
        h=(1.0-phi0**2)**2
        #h=(1.0+phi0)*0.5
        g=(2.0-phi0)/3.0
        PHI_POW=3
        residual = [
            ( ufl.inner(phi,v_phi) * ufl.dx - ufl.inner(phi0,v_phi) * ufl.dx + dt*(1/M0)*ufl.inner(matrixT*ufl.grad(mu_mid),ufl.grad(v_phi)) * ufl.dx
                -dt*nu*ufl.inner((n-tresh)*h, v_phi) * ufl.dx
            ),
            (
                ufl.inner(mu, v_mu) * ufl.dx
                -k* ufl.inner(phi**PHI_POW, v_mu) * ufl.dx
                -k* ufl.inner(-phi0, v_mu) * ufl.dx
                - (epsilon*epsilon) * ufl.inner(ufl.grad(phi), ufl.grad(v_mu)) * ufl.dx
            ),
            (
                ufl.inner(n, v_n) * ufl.dx - ufl.inner(n0, v_n) * ufl.dx + dt * Dn* ufl.inner(matrixD*ufl.grad(n),ufl.grad(v_n)) * ufl.dx
                - dt * Sn * ufl.inner((1.0-n)*g,v_n) * ufl.dx + deltan* dt *ufl.inner(n*h,v_n)*ufl.dx
            )

        ]
        self._residual = residual
        self._residual_cpp = dolfinx.fem.form(residual)
        # Define jacobian block form of the problem
        jacobian = [
            [ufl.derivative(residual[0], phi, dphi), ufl.derivative(residual[0], mu, dmu), ufl.derivative(residual[0], n, dn)],
            [ufl.derivative(residual[1], phi, dphi), ufl.derivative(residual[1], mu, dmu), ufl.derivative(residual[1], n, dn)],
            [ufl.derivative(residual[2], phi, dphi), ufl.derivative(residual[2], mu, dmu), ufl.derivative(residual[2], n, dn)]
        ]
        self._jacobian = jacobian
        self._jacobian_cpp = dolfinx.fem.form(jacobian)

    @property
    def dt(self) -> petsc4py.PETSc.RealType:
        """Return the time step."""
        return self._dt

    @property
    def T(self) -> petsc4py.PETSc.RealType:
        """Return the final time."""
        return self._T

    @property
    def function_space_phi(self) -> dolfinx.fem.FunctionSpace:
        """Return the function spaces of the problem."""
        return self._function_space_phi

    @property
    def function_space_mu(self) -> dolfinx.fem.FunctionSpace:
        """Return the function spaces of the problem."""
        return self._function_space_mu

    @property
    def function_space_n(self) -> dolfinx.fem.FunctionSpace:
        """Return the function spaces of the problem."""
        return self._function_space_n

    @property
    def function_spaces(self) -> typing.Tuple[dolfinx.fem.FunctionSpace, dolfinx.fem.FunctionSpace, dolfinx.fem.FunctionSpace]:
        """Return the function spaces of the problem."""
        return self._V

    @property
    def trials_and_tests(self) -> typing.Tuple[ufl.Argument, ufl.Argument, ufl.Argument, ufl.Argument]:
        """Return the UFL arguments used in the construction of the forms."""
        return self._trial_dphi, self._trial_dmu, self._trial_dn, self._test_v_phi, self._test_v_mu, self._test_v_n

    @property
    def solution(self) -> typing.Tuple[dolfinx.fem.Function, dolfinx.fem.Function, dolfinx.fem.Function]:
        """Return the components of the solution of the problem for the latest parameter value."""
        return self._solutions

    @property
    def solution0(self) -> typing.Tuple[dolfinx.fem.Function, dolfinx.fem.Function, dolfinx.fem.Function]:
        """Return the components of the solution of the problem for the latest parameter value."""
        return self._solutions0

    @property
    def residual_form(self) -> typing.List[ufl.Form]:  # type: ignore[no-any-unimported]
        """Return the residual block form of the problem."""
        return self._residual

    @property
    def jacobian_form(self) -> typing.List[typing.List[ufl.Form]]:  # type: ignore[no-any-unimported]
        """Return the jacobian block form of the problem."""
        return self._jacobian

    @property
    def e_symb(self) -> rbnicsx.backends.SymbolicParameters:
        """Return the symbolic parameters of the problem."""
        return self._e_symb

    def update_solutions(self, x: petsc4py.PETSc.Vec) -> None:  # type: ignore[no-any-unimported]
        """Update `self._solutions` with data in `x`."""
        x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(x, [self._V[0].dofmap, self._V[1].dofmap, self._V[2].dofmap]) as x_wrapper:
            for x_wrapper_local, sub_solution in zip(x_wrapper, self._solutions):
                with sub_solution.vector.localForm() as sub_solution_local:
                    sub_solution_local[:] = x_wrapper_local

    def _assemble_residual(  # type: ignore[no-any-unimported]
        self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, residual_vec: petsc4py.PETSc.Vec
    ) -> None:
        """Assemble the residual."""
        self.update_solutions(x)
        with residual_vec.localForm() as residual_vec_local:
            residual_vec_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector_block(  # type: ignore[misc]
            residual_vec, self._residual_cpp, self._jacobian_cpp, [], x0=x, scale=-1.0)

    def _assemble_jacobian(  # type: ignore[no-any-unimported]
        self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, jacobian_mat: petsc4py.PETSc.Mat,
        preconditioner_mat: petsc4py.PETSc.Mat
    ) -> None:
        """Assemble the jacobian."""
        jacobian_mat.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_block(  # type: ignore[misc]
            jacobian_mat, self._jacobian_cpp, [], diagonal=1.0)  # type: ignore[arg-type]
        jacobian_mat.assemble()

    def solve(self, e: np.typing.NDArray[np.float64], plot: bool) -> typing.Tuple[rbnicsx.backends.FunctionsList, rbnicsx.backends.FunctionsList, rbnicsx.backends.FunctionsList, typing.List[np.float64]]:
        """Assign the provided parameter value to boundary conditions and solve the problem."""
        for s in (0, 1, 2):
            with self._solutions0[s].vector.localForm() as sub_solution0_local:
                sub_solution0_local[:] = np.zeros_like(sub_solution0_local)
        self._e_symb.value[:] = e
        self.set_ic()
        K = int(self._T/self._dt)
        # Define time list and solution function lists
        times = []
        solution_phi = rbnicsx.backends.FunctionsList(self._V[0])
        solution_n = rbnicsx.backends.FunctionsList(self._V[1])
        solution_mu = rbnicsx.backends.FunctionsList(self._V[2])

        for i in range(1, K+1):
            # Compute the current time
            t = i*self._dt
            times.append(t)
            if MPI.COMM_WORLD.rank==0:
                print("t = %5.4f" % t, "      i =", i)
            app_phi, app_mu, app_n = self._solve()
            solution_phi.append(app_phi)
            solution_mu.append(app_mu)
            solution_n.append(app_n)
            if plot:
                filephi.write_function(app_phi, t)
                filemu.write_function(app_mu, t)
                filen.write_function(app_n, t)

            # Store the solution in up_prev
            for s in (0, 1, 2):
                with self._solutions0[s].vector.localForm() as sub_solution0_local, self._solutions[s].vector.localForm() as sub_solution_local:
                    sub_solution0_local[:] = sub_solution_local[:]
        return (solution_phi, solution_mu, solution_n, times)

    def set_ic(self):
        #self._solutions0[0].interpolate(lambda x: 2*np.exp(-(L_car**2/10)*(((x[0]-50.0/L_car)**2+(x[1]-50.0/L_car)**2))**2)-1)
        #filephi.write_function(self._solutions0[0], t)
        self._solutions0[0].interpolate(lambda x : 1.8*np.exp(-1e+2*(((x[0]-19.3)**2+(x[1]-30.8)**2+(x[2]-3.0)**2)**2))-1)
        return

    def _solve(self) -> typing.Tuple[dolfinx.fem.Function, dolfinx.fem.Function,dolfinx.fem.Function]:
        """Solve the nonlinear problem with SNES."""
        snes = petsc4py.PETSc.SNES().create(mesh.comm)
        snes.setTolerances(max_it=20)
        snes.getKSP().setType("preonly")
        snes.getKSP().getPC().setType("lu")
        snes.getKSP().getPC().setFactorSolverType("mumps")
        residual_vec = dolfinx.fem.petsc.create_vector_block(self._residual_cpp)
        snes.setFunction(self._assemble_residual, residual_vec)
        jacobian_mat = dolfinx.fem.petsc.create_matrix_block(self._jacobian_cpp)
        snes.setJacobian(self._assemble_jacobian, J=jacobian_mat, P=None)
        if MPI.COMM_WORLD.rank==0:
            snes.setMonitor(lambda _, it, residual: print(it, residual))
        solution = dolfinx.fem.petsc.create_vector_block(self._residual_cpp)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(solution, [self._V[0].dofmap, self._V[1].dofmap, self._V[2].dofmap]) as x_wrapper:
                for x_wrapper_local, sub_solution in zip(x_wrapper, self._solutions):
                    with sub_solution.vector.localForm() as sub_solution_local:
                        x_wrapper_local[:] = sub_solution_local

        snes.solve(None, solution)
        self.update_solutions(solution)
        app_phi = self._solutions[0].copy()
        app_mu = self._solutions[1].copy()
        app_n = self._solutions[2].copy()
        app_phi.name = "phi"
        app_mu.name = "mu"
        app_n.name = "n"
        return (app_phi, app_mu, app_n)


# In[2]:


# In[4]:


class ProjectionBasedReducedProblem(abc.ABC):
    """Define a linear projection-based problem, and solve it with KSP."""

    def __init__(self, problem: Problem) -> None:
        self._problem = problem
        # Define velocity basis functions storage
        basis_functions_phi = rbnicsx.backends.FunctionsList(self._problem.function_spaces[0])
        self._basis_functions_phi = basis_functions_phi
        basis_functions_mu = rbnicsx.backends.FunctionsList(self._problem.function_spaces[1])
        self._basis_functions_mu = basis_functions_mu
        basis_functions_n = rbnicsx.backends.FunctionsList(self._problem.function_spaces[2])
        self._basis_functions_n = basis_functions_n
        # Get trial and test functions from the high fidelity problem
        dphi, dmu, dn, v_phi, v_mu, v_n = self._problem.trials_and_tests

        #multiphenicsx.io.plot_scalar_field(basis_functions_phi[0], "b0")

        inner_product_phi = ufl.inner(dphi, v_phi) * ufl.dx
        self._inner_product_phi = inner_product_phi
        self._inner_product_action_phi = rbnicsx.backends.bilinear_form_action(inner_product_phi, part="real")

        inner_product_mu = ufl.inner(dmu, v_mu) * ufl.dx
        self._inner_product_mu = inner_product_mu
        self._inner_product_action_mu = rbnicsx.backends.bilinear_form_action(inner_product_mu, part="real")
        inner_product_n = ufl.inner(dn, v_n) * ufl.dx
        self._inner_product_n = inner_product_n
        self._inner_product_action_n = rbnicsx.backends.bilinear_form_action(inner_product_n, part="real")
        # Define the forms for the solve
        self._residual_cpp = dolfinx.fem.form(self._assemble_residual)
        self._jacobian_cpp = dolfinx.fem.form(self._assemble_jacobian)

        self._e_symb = self._problem.e_symb
        self._e = np.zeros(self._e_symb.value.shape)

    @property
    def basis_functions_phi(self) -> rbnicsx.backends.FunctionsList:
        """Return the velocity basis functions of the reduced problem."""
        return self._basis_functions_phi

    @property
    def basis_functions_mu(self) -> rbnicsx.backends.FunctionsList:
        """Return the supremizer basis functions of the reduced problem."""
        return self._basis_functions_mu

    @property
    def basis_functions_n(self) -> rbnicsx.backends.FunctionsList:
        """Return the pressure basis functions of the reduced problem."""
        return self._basis_functions_n

    @property
    def inner_product_form_phi(self) -> ufl.Form:
        return self._inner_product_phi

    @property
    def inner_product_form_mu(self) -> ufl.Form:
        return self._inner_product_mu

    @property
    def inner_product_form_n(self) -> ufl.Form:
        return self._inner_product_n

    @property
    def inner_product_action_phi(self) -> typing.Callable:
        return self._inner_product_action_phi

    @property
    def inner_product_action_mu(self) -> typing.Callable:
        return self._inner_product_action_mu

    @property
    def inner_product_action_n(self) -> typing.Callable:
        return self._inner_product_action_n

    def update_solutions(self, x: petsc4py.PETSc.Vec) -> None:
        try:
            N = (x.size)//3
            with rbnicsx.online.BlockVecSubVectorCopier(x, [N, N, N]) as x_copier:
                x_phi, x_mu, x_n = [xx for xx in x_copier]
            reconstructed_solutions = self._reconstruct_solution(x_phi, x_mu, x_n)
            for s in (0, 1, 2):
                with reconstructed_solutions[s].vector.localForm() as reconstructed_solution_local, self._problem.solution[s].vector.localForm() as sub_solution_local:
                    sub_solution_local[:] = reconstructed_solution_local[:]

        except Exception as e:
            print(e)

    @abc.abstractmethod
    def _assemble_residual(
        self, t: petsc4py.PETSc.RealType
    ) -> None:
        """Assemble the residual vector of dimension N."""
        pass

    @abc.abstractmethod
    def _assemble_jacobian(
        self, t: petsc4py.PETSc.RealType
    ) -> None:
        """Assemble the jacobian matrix of dimension N."""
        pass

    def solve(self, e: np.typing.NDArray[np.float64], N: int) -> typing.Tuple[rbnicsx.online.FunctionsList, rbnicsx.online.FunctionsList, rbnicsx.online.FunctionsList, typing.List[np.float64]]:
        """Apply shape parametrization and solve the problem."""
       # self._problem.solution0[0].interpolate(lambda x: 2*np.exp(-(L_car**2/10)*(((x[0]-50.0/L_car)**2+(x[1]-50.0/L_car)**2))**2)-1)

        self._e[:] = e
        self._e_symb.value[:] = e
        K = int(problem.T/problem.dt)
        self._reduced_phi_prev = None
        self._reduced_mu_prev = None
        self._reduced_n_prev = None
        # Define time list and solution function lists
        reduced_times = []
        reduced_solution_phi = rbnicsx.online.FunctionsList(N)
        reduced_solution_mu = rbnicsx.online.FunctionsList(N)
        reduced_solution_n = rbnicsx.online.FunctionsList(N)
        self.set_ic()
        self._dt=problem.dt
        for i in range(1, K+1):
            # Compute the current time
            t = i*self._dt
            reduced_times.append(t)
            if MPI.COMM_WORLD.rank==0:
                print("t = %5.4f" % t, "      i =", i)
            phi, mu, n = self._solve(t, N)
            self._reduced_phi_prev = phi
            self._reduced_mu_prev = mu
            self._reduced_n_prev = n
#            if MPI.COMM_WORLD.rank==0:
#                print(phi[:])
            reduced_solution_phi.append(phi)
            reduced_solution_mu.append(mu)
            reduced_solution_n.append(n)
            # Store the solution in up_prev
            reconstructed_solutions = self._reconstruct_solution(phi, mu, n)
            for s in (0, 1, 2):
                with reconstructed_solutions[s].vector.localForm() as reconstructed_solution_local, self._problem.solution0[s].vector.localForm() as sub_solution0_local:
                    sub_solution0_local[:] = reconstructed_solution_local[:]
        return (reduced_solution_phi, reduced_solution_mu, reduced_solution_n, reduced_times)

    def _solve(self,t: petsc4py.PETSc.RealType, N: int) -> typing.Tuple[petsc4py.PETSc.Vec, petsc4py.PETSc.Vec,petsc4py.PETSc.Vec]:
        """Solve the nonlinear online problem with SNES."""
        snes = petsc4py.PETSc.SNES().create(mesh.comm)
        snes.setTolerances(max_it=20)
        snes.getKSP().setType("preonly")
        snes.getKSP().getPC().setType("lu")
        residual_vec = rbnicsx.online.create_vector_block([N, N, N])
        snes.setFunction(self._assemble_residual(t), residual_vec)

        jacobian_mat = rbnicsx.online.create_matrix_block([N, N, N], [N, N, N])

        snes.setJacobian(self._assemble_jacobian(t), J=jacobian_mat, P=None)

        if MPI.COMM_WORLD.rank==0:
            snes.setMonitor(lambda _, it, residual: print("Iteration",it, residual))
        reduced_solution = rbnicsx.online.create_vector_block([N, N, N])
        if self._reduced_phi_prev is not None:
            with rbnicsx.online.BlockVecSubVectorWrapper(reduced_solution, [N, N, N]) as x_wrapper:
                    for x_wrapper_local, sub_solution in zip(x_wrapper, (self._reduced_phi_prev, self._reduced_mu_prev, self._reduced_n_prev)):
                        x_wrapper_local[:] = sub_solution
        snes.solve(None, reduced_solution)

        if MPI.COMM_WORLD.rank==0:
            print("Converged: ", snes.converged)
        with rbnicsx.online.BlockVecSubVectorCopier(reduced_solution, [N, N, N]) as reduced_solution_copier:
            return [reduced_solution for reduced_solution in reduced_solution_copier]

    def reconstruct_solution(self, reduced_solution_phi: rbnicsx.online.FunctionsList, reduced_solution_mu:rbnicsx.online.FunctionsList,
                             reduced_solution_n: rbnicsx.online.FunctionsList
                             ) -> typing.Tuple[rbnicsx.backends.FunctionsList, rbnicsx.backends.FunctionsList, rbnicsx.backends.FunctionsList]:
        reconstructed_phi = rbnicsx.backends.FunctionsList(self._problem.function_spaces[0])
        reconstructed_mu = rbnicsx.backends.FunctionsList(self._problem.function_spaces[1])
        reconstructed_n = rbnicsx.backends.FunctionsList(self._problem.function_spaces[2])
        for i in range(0, len(reduced_solution_phi)):
            """Reconstructed velocity reduced solution on the high fidelity space."""
            phi_app, mu_app, n_app = self._reconstruct_solution(reduced_solution_phi[i], reduced_solution_mu[i], reduced_solution_n[i])
            phi_app.name   = "r_phi"
            mu_app.name   = "r_mu"
            n_app.name   = "r_n"
            filephi_red.write_function(phi_app, i*self._dt)
            filemu_red.write_function(mu_app, i*self._dt)
            filen_red.write_function(n_app, i*self._dt)
            reconstructed_phi.append(phi_app)
            reconstructed_mu.append(mu_app)
            reconstructed_n.append(n_app)
        return (reconstructed_phi, reconstructed_mu, reconstructed_n)

    def _reconstruct_solution(self, reduced_solution_phi: petsc4py.PETSc.Vec, reduced_solution_mu:
                             petsc4py.PETSc.Vec, reduced_solution_n:
                             petsc4py.PETSc.Vec) -> typing.Tuple[dolfinx.fem.Function, dolfinx.fem.Function, dolfinx.fem.Function]:
        N_phi = reduced_solution_phi.size
        N_mu = reduced_solution_mu.size
        N_n = reduced_solution_n.size
        phi_app   = dolfinx.fem.Function(self._problem.function_spaces[0])
        mu_app   = dolfinx.fem.Function(self._problem.function_spaces[1])
        n_app   = dolfinx.fem.Function(self._problem.function_spaces[2])
        #print(len(self.basis_functions_phi),N_phi,reduced_solution_phi.size)
        phi_app   = self.basis_functions_phi[:N_phi] * reduced_solution_phi
        mu_app   = self.basis_functions_mu[:N_mu] * reduced_solution_mu
        n_app   = self.basis_functions_n[:N_n] * reduced_solution_n

        return (phi_app, mu_app, n_app)

    def compute_norm_phi(self, function: dolfinx.fem.Function) -> petsc4py.PETSc.RealType:
        """Compute the norm of a function using the natural inner product for the velocity."""
        return np.sqrt(self._inner_product_action_phi(function)(function))

    def compute_norm_mu(self, function: dolfinx.fem.Function) -> petsc4py.PETSc.RealType:
        """Compute the norm of a function using the natural inner product for the pressure."""
        return np.sqrt(self._inner_product_action_mu(function)(function))

    def compute_norm_n(self, function: dolfinx.fem.Function) -> petsc4py.PETSc.RealType:
        """Compute the norm of a function using the natural inner product for the pressure."""
        return np.sqrt(self._inner_product_action_(function)(function))

    def set_ic(self):
        #self._problem.solution0[0].interpolate(lambda x: 2*np.exp(-(L_car**2/10)*(((x[0]-50.0/L_car)**2+(x[1]-50.0/L_car)**2))**2)-1)
        self._problem.solution0[0].interpolate(lambda x : 1.8*np.exp(-1e+2*(((x[0]-19.3)**2+(x[1]-30.8)**2+(x[2]-3.0)**2)**2))-1)

        #filephi.write_function(self._solutions0[0], t)
        return


# In[5]:


class ProjectionBasedInefficientReducedProblem(ProjectionBasedReducedProblem):
    """Specialize the projection-based problem to inefficient assembly of the system."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        # Store forms of the problem
        dphi, dmu, dn, v_phi, v_mu, v_n = self._problem.trials_and_tests
        f_action = [None, None, None]
        f_action[0] = rbnicsx.backends.linear_form_action(problem.residual_form[0])
        f_action[1] = rbnicsx.backends.linear_form_action(problem.residual_form[1])
        f_action[2] = rbnicsx.backends.linear_form_action(problem.residual_form[2])
        a_action = [[None, None, None], [None, None, None], [None, None, None]]
        a_action[0][0] = rbnicsx.backends.bilinear_form_action(problem.jacobian_form[0][0])
        a_action[1][0] = rbnicsx.backends.bilinear_form_action(problem.jacobian_form[1][0])
        a_action[2][0] = rbnicsx.backends.bilinear_form_action(dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0)) * ufl.inner(dn,v_phi) * ufl.dx)
        a_action[0][1] = rbnicsx.backends.bilinear_form_action(problem.jacobian_form[0][1])
        a_action[1][1] = rbnicsx.backends.bilinear_form_action(problem.jacobian_form[1][1])
        a_action[2][1] = rbnicsx.backends.bilinear_form_action(dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0)) * ufl.inner(dn,v_mu) * ufl.dx)
        a_action[0][2] = rbnicsx.backends.bilinear_form_action(problem.jacobian_form[0][2])
        a_action[1][2] = rbnicsx.backends.bilinear_form_action(dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0)) * ufl.inner(dmu,v_n) * ufl.dx)
        a_action[2][2] = rbnicsx.backends.bilinear_form_action(problem.jacobian_form[2][2])
        self._f_action = f_action
        self._a_action = a_action

    def _assemble_residual(
        self, t: petsc4py.PETSc.RealType
    ) -> typing.Callable:
        def _assemble_residual_t(
            snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, residual_vec: petsc4py.PETSc.Vec
        ) -> None:
            try:
                """Assemble the residual."""

                N = (x.size)//3
                #if MPI.COMM_WORLD.rank==0:
                #    print(x[:])
                print(MPI.COMM_WORLD.rank, x.array)
                self.update_solutions(x)
                residual_vec.set(0.0)
                rbnicsx.backends.project_vector_block(residual_vec, self._f_action, [self._basis_functions_phi[:N], self._basis_functions_mu[:N], self._basis_functions_n[:N]])
                #residual_vec.view()
            except Exception as e:
                print(e)
        return _assemble_residual_t

    def _assemble_jacobian(
        self, t: petsc4py.PETSc.RealType
    ) -> typing.Callable:
        def _assemble_jacobian_t(
            snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, jacobian_mat: petsc4py.PETSc.Mat,
            preconditioner_mat: petsc4py.PETSc.Mat
        ) -> None:
            try:
                """Assemble the jacobian."""
                N = (x.size)//3
                jacobian_mat.zeroEntries()
                rbnicsx.backends.project_matrix_block(jacobian_mat, self._a_action,
                                                            [self._basis_functions_phi[:N], self._basis_functions_mu[:N],
                                                             self._basis_functions_n[:N]])

                jacobian_mat.assemble()
                #jacobian_mat.view()
            except Exception as e:
                print(e)
        return _assemble_jacobian_t


# In[6]:


class PODGInefficientReducedProblem(ProjectionBasedInefficientReducedProblem):
    """Specialize the inefficient projection-based problem to basis generation with POD."""

    pass


t=0
problem=Problem()


### GENERETE THE TRAINING SET ####

def generate_training_set() -> np.typing.NDArray[np.float64]:
    """Generate an equispaced training set using numpy."""
    training_set_0 = np.linspace(0.1, 0.4, 2)   #np.linspace(800.0, 1800.00, 2)
    training_set_1 = np.linspace(2000.0, 4000.0, 2)
    training_set_2 = np.linspace(600.0, 900.0, 2)    #np.linspace(0.15, 0.4, 2)
    training_set_3 = np.linspace(0.2, 0.3, 2)
    training_set_4 = np.linspace(5.0e+03, 5.0e+04, 2)    #np.linspace(10000.0, 1.0e+06, 2)
    training_set_5 = np.linspace(5.0e+03, 5.0e+04, 2)         #np.linspace(0.03, 0.73, 2)
   # training_set_6 = np.linspace(5.0e+04, 5.0e+04, 2)    #np.linspace(10000.0, 1.0e+06, 2)
   # training_set_7 = np.linspace(21, 30, 3)
    training_set = np.array(list(itertools.product(training_set_5, training_set_1, training_set_2, training_set_3, training_set_4, training_set_0)))
    return training_set


training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)
Nint = 30
Nmax = 40


if MPI.COMM_WORLD.rank==0:
    print(rbnicsx.io.TextBox("POD-Galerkin offline phase begins", fill="="))
    print("")

    print("set up snapshots matrix")
snapshots_matrix_phi = rbnicsx.backends.FunctionsList(problem.function_spaces[0])
snapshots_matrix_mu = rbnicsx.backends.FunctionsList(problem.function_spaces[0])
snapshots_matrix_n = rbnicsx.backends.FunctionsList(problem.function_spaces[0])

if MPI.COMM_WORLD.rank==0:
    print("set up reduced problem")
reduced_problem = PODGInefficientReducedProblem(problem)
red_prob=PODGInefficientReducedProblem(problem)
if MPI.COMM_WORLD.rank==0:
    print("")


### LOOP OVER THE MODES GENERETED BY THE SPLITTED SCRIPT ###

for id in range(4):
    matrix_phi = rbnicsx.backends.FunctionsList(problem.function_space_phi)
    matrix_mu = rbnicsx.backends.FunctionsList(problem.function_space_mu)
    matrix_n = rbnicsx.backends.FunctionsList(problem.function_space_n)
    matrix_phi.load(f"modes_{serial_or_parallel}/modes_"+str(id), "modes_phi")
    matrix_mu.load(f"modes_{serial_or_parallel}/modes_"+str(id), "modes_mu")
    matrix_n.load(f"modes_{serial_or_parallel}/modes_"+str(id), "modes_n")

    for b in (matrix_phi, matrix_mu, matrix_n):
        for i in range(Nint):
            b[i].vector.ghostUpdate(
                addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)


    if MPI.COMM_WORLD.rank==0:
        print("update snapshots matrix")
    snapshots_matrix_phi.extend(matrix_phi)
    snapshots_matrix_mu.extend(matrix_mu)
    snapshots_matrix_n.extend(matrix_n)

eigenvalues, modes, _ = rbnicsx.backends.proper_orthogonal_decomposition_block(
    [snapshots_matrix_phi, snapshots_matrix_mu, snapshots_matrix_n], [reduced_problem.inner_product_action_phi, reduced_problem.inner_product_action_mu, reduced_problem.inner_product_action_n], N=Nmax, tol=0.0)
eigenvalues_phi, eigenvalues_mu, eigenvalues_n = eigenvalues
modes_phi, modes_mu, modes_n = modes

reduced_problem.basis_functions_phi.extend(modes_phi)
reduced_problem.basis_functions_mu.extend(modes_mu)
reduced_problem.basis_functions_n.extend(modes_n)
reduced_problem.basis_functions_phi.save(f"basis_functions_{serial_or_parallel}", "basis_functions_phi")
reduced_problem.basis_functions_mu.save(f"basis_functions_{serial_or_parallel}", "basis_functions_mu")
reduced_problem.basis_functions_n.save(f"basis_functions_{serial_or_parallel}", "basis_functions_n")
toc=time.time()
print("Elapsed time ", toc-tic)

positive_eigenvalues_phi = np.where(eigenvalues_phi > 0., eigenvalues_phi, np.nan)
singular_values_phi = np.sqrt(positive_eigenvalues_phi)

positive_eigenvalues_mu = np.where(eigenvalues_mu > 0., eigenvalues_mu, np.nan)
singular_values_mu = np.sqrt(positive_eigenvalues_mu)

positive_eigenvalues_n = np.where(eigenvalues_n > 0., eigenvalues_n, np.nan)
singular_values_n = np.sqrt(positive_eigenvalues_n)

fig = go.Figure()
fig.add_scatter(
    x=np.arange(singular_values_phi.shape[0]), y=singular_values_phi / singular_values_phi[0], mode="markers+lines")
fig.update_layout(title="Normalized singular values for phi")
fig.update_xaxes(title_text="N"),
fig.update_yaxes(title_text=r"$\frac{\sigma_N}{\sigma_0}$", type="log", exponentformat="power")
fig.write_image(f"snapshots_{serial_or_parallel}/singulare_values_phi.png")

fig = go.Figure()
fig.add_scatter(
    x=np.arange(singular_values_mu.shape[0]), y=singular_values_mu / singular_values_mu[0], mode="markers+lines")
fig.update_layout(title="Normalized singular values for mu")
fig.update_xaxes(title_text="N"),
fig.update_yaxes(title_text=r"$\frac{\sigma_N}{\sigma_0}$", type="log", exponentformat="power")
fig.write_image(f"snapshots_{serial_or_parallel}/singulare_values_mu.png")

fig = go.Figure()
fig.add_scatter(
    x=np.arange(singular_values_n.shape[0]), y=singular_values_n / singular_values_n[0], mode="markers+lines")
fig.update_layout(title="Normalized singular values for n")
fig.update_xaxes(title_text="N"),
fig.update_yaxes(title_text=r"$\frac{\sigma_N}{\sigma_0}$", type="log", exponentformat="power")
fig.write_image(f"snapshots_{serial_or_parallel}/singulare_values_n.png")

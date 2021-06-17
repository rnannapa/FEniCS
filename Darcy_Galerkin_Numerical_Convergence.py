"""
Formulation: Galerkin formulation 
             
Interpolation: The code can handle both P2P0 and P1P1 

Problem: Numerical convergence problem 

Written By: Professor Kalyana Babu Nakshatrala 
            University of Houston

Edited By:  Raj Gopal Nannapaneni 
            University of Houston 
"""
from dolfin import *
import numpy as np
from math import * 

#===========================;
#  Load computational mesh  ;
#===========================;
nx = 10
mesh = UnitSquareMesh(nx,nx)

#===================================;
#  Define function spaces for P2P0  ;
#===================================;
V = VectorFunctionSpace(mesh,"CG",2)
Q = FunctionSpace(mesh,"DG",0)
W = V * Q

#===================================;
#  Define function spaces for P1P1  ;
#===================================;
# V = VectorFunctionSpace(mesh,"CG",1)
# Q = FunctionSpace(mesh,"CG",1)
# W = V * Q

#===========================;
#  Drag coefficient (mu/k)  ;
#===========================;
alpha = Constant(1.0)

#=====================;
#  Define boundaries  ;
#=====================;
def top(x, on_boundary): return near(x[1],1.0)

def right(x, on_boundary): return near(x[0],1.0)

def left(x, on_boundary): return near(x[0],0.0)

def bottom(x, on_boundary): return near(x[1],0.0) 

# No-slip boundary condition for velocity
noslipBC = Constant(0.0)

bc_B = DirichletBC(W.sub(0).sub(1),noslipBC,bottom)
bc_L = DirichletBC(W.sub(0).sub(0),noslipBC,left)
bc_R = DirichletBC(W.sub(0).sub(0),noslipBC,right)
bc_T = DirichletBC(W.sub(0).sub(1),noslipBC,top)

# Collect boundary conditions
bcs = [bc_B,bc_L,bc_R,bc_T]

#==============================;
#  Expression for body force   ;
#==============================;
f = Expression(("sin(pi * x[0]) * cos(pi * x[1]) + 2 * x[0] * x[1] * x[1]","-cos(pi * x[0]) * sin(pi * x[1]) + 2 * x[0] * x[0] * x[1]"))

#==============================;
#  Define variational problem  ;
#==============================;
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

stab_param = 0.0001;

a = dot(alpha*u,v) * dx - div(v)*p*dx - q*div(u)*dx + stab_param * p * q * dx
L = inner(f,v)*dx
outfileV = File('Numerical_Convergence_Velocity.pvd')
outfileP = File('Numerical_Convergence_Pressure.pvd')
#=================================;
#  Solve the resulting equations  ;
#=================================;
U = Function(W)
solve(a == L, U, bcs)

# Get sub-functions
u, p = U.split()
outfileV << u
outfileP << p

# Plot solution
# plot(u,title="Velocity")
# plot(p,title="Pressure")
# interactive()

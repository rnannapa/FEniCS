"""
Formulation: Galerkin formulation 
             
Interpol1ation: The code can handle both P2P0 and P1P1

Problem: 2D Constant Flow 

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
# V = VectorFunctionSpace(mesh,"CG",2)
# Q = FunctionSpace(mesh,"DG",0)
# W = V * Q

#===================================;
#  Define function spaces for P1P1  ;
#===================================;
V = VectorFunctionSpace(mesh,"CG",1)
Q = FunctionSpace(mesh,"CG",1)
W = V * Q

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

def point(x, on_boundary): return x[0]==1.0 and x[1] == 1.0

# No-slip boundary condition for velocity
bc_B = DirichletBC(W.sub(0).sub(1),Constant(0.0),bottom)
bc_T = DirichletBC(W.sub(0).sub(1),Constant(0.0),top)
bc_L = DirichletBC(W.sub(0).sub(0),Constant(+1.0),left)
bc_R = DirichletBC(W.sub(0).sub(0),Constant(+1.0),right)

# Pressure Boundary Conditon
bc_TC = DirichletBC(W.sub(1),Constant(+1.0),point,method = "pointwise")


# Collect boundary conditions
bcs = [bc_B,bc_L,bc_R,bc_T,bc_TC]

#==============================;
#  Expression for body force   ;
#==============================;
f = Constant((0.0,0.0))

#==============================;
#  Define variational problem  ;
#==============================;
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

stab_param = 1e-3;

a = dot(alpha*u,v) * dx - div(v)*p*dx - q*div(u)*dx + stab_param * p * q * dx
L = inner(f,v)*dx

# a = dot(alpha*u,v) * dx + dot(v,grad(p))*dx - q*div(u)*dx - \
# 	0.5 * dot(alpha * v + grad(q),1.0/alpha * (alpha * u + grad(p))) * dx
# L = inner(f,v) *dx - 0.5 * dot(alpha * v + grad(q),1.0/alpha * f) * dx

#=================================;
#  Solve the resulting equations  ;
#=================================;

outfileV = File('ConstantFlow_2D_Velocity.pvd')
outfileP = File('ConstantFlow_2D_Pressure.pvd')
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

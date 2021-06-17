"""
Formulation: Galerkin formulation 
             
Interpol1ation: The code can handle both P2P0 and P1P1

Problem: 3D Constant Flow 

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
#======
n=10
lx = 5 
ly = 5 
lz = 5
mesh = BoxMesh(Point(0.0,0.0,0.0),Point(lx,ly,lz),n,n,n)

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
def left(x, on_boundary): return near(x[0],0.0)

def bottom(x, on_boundary): return near(x[1],0.0)

def back(x, on_boundary): return near(x[2],0.0)

def right(x, on_boundary): return near(x[0],lx)

def top(x, on_boundary): return near(x[1],ly)

def front(x, on_boundary): return near(x[2],lz)

def Corner(x, on_boundary): return x[0] == 0.0 and x[1] == 0.0 and x[2] == 0.0


# No-slip boundary condition for velocity
bc_Left 	= DirichletBC(W.sub(0).sub(0),Constant(+1.0),left)
bc_Right 	= DirichletBC(W.sub(0).sub(0),Constant(+1.0),right)
bc_Bottom 	= DirichletBC(W.sub(0).sub(1),Constant(0.0),bottom)
bc_Top 		= DirichletBC(W.sub(0).sub(1),Constant(0.0),top)
bc_Front 	= DirichletBC(W.sub(0).sub(2),Constant(0.0),front)
bc_Back 	= DirichletBC(W.sub(0).sub(2),Constant(0.0),back)

# Pressure Boundary Conditon
bc_Corner = DirichletBC(W.sub(1),Constant(+0.0),Corner,method = "pointwise")


# Collect boundary conditions
bcs = [bc_Bottom,bc_Left,bc_Right,bc_Top,bc_Front,bc_Back,bc_Corner]

#==============================;
#  Expression for body force   ;
#==============================;
f = Constant((0.0,0.0,0.0))

#==============================;
#  Define variational problem  ;
#==============================;
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

stab_param = 1e-3;

a = dot(alpha*u,v) * dx - div(v)*p*dx - q*div(u)*dx + stab_param * p * q * dx
L = inner(f,v)*dx
# a = dot(alpha*u,v) * dx + dot(v,grad(p))*dx - q*div(u)*dx - \
# 	0.5 * dot(alpha * v + grad(q),1.0/alpha * (alpha * u + grad(p))) * dx #+ stab_param * p * q * dx
# L = inner(f,v) *dx - 0.5 * dot(alpha * v + grad(q),1.0/alpha * f) * dx

#=================================;
#  Solve the resulting equations  ;
#=================================;

outfileV = File('ConstantFlow_3D_Velocity.pvd')
outfileP = File('ConstantFlow_3D_Pressure.pvd')
U = Function(W)
solve(a == L, U, bcs)

# Get sub-functions
u, p = U.split()

outfileV << u
outfileP << p


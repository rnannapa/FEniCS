"""
Formulation: VMS formulation 
             
Interpolation: P1P1 

Comment: Only P1P1 should be enough for this problem. 
         There is no need for P2P0. If you want, you 
         can try P2P0 also.

Problem: 1D Problem 

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
mesh = UnitIntervalMesh(nx)

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
def right(x, on_boundary): return near(x[0],1.0) 

#  Pressure boundary condition  
bc_L = DirichletBC(W.sub(1),Constant(1.0),left)
bc_R = DirichletBC(W.sub(1),Constant(0.0),right)

# Collect boundary conditions
bcs = [bc_L,bc_R]

#==============================;
#  Expression for body force   ;
#==============================;
f = Constant((0.0,))

#==============================;
#  Define variational problem  ;
#==============================;
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

a = dot(alpha*u,v) * dx + dot(v,grad(p))*dx - q*div(u)*dx - \
	0.5 * dot(alpha * v + grad(q),1.0/alpha * (alpha * u + grad(p))) * dx
L = inner(f,v) *dx - 0.5 * dot(alpha * v + grad(q),1.0/alpha * f) * dx

#=================================;
#  Solve the resulting equations  ;
#=================================;
# outfileVelocity = File('VMS_1D_Velocity.pvd')
# outfilePressure = File('VMS_1D_Pressure.pvd')

U = Function(W)
solve(a == L, U, bcs)

# Get sub-functions
u, p = U.split()

#  Print velocity and pressure 
velocity = interpolate(u,V)
print "Velocity = ", velocity.vector().array()

pressure = interpolate(p,Q)
print "Pressure = ", pressure.vector().array()


#=================;
#  Plot solution  ;
#=================;
# outfileVelocity << velocity
# outfilePressure << pressure

"""
Formulation: Galerkin formulation 
             
Interpolation: The code can handle both P2P0 and P1P1 

Problem: Reservoir Problem  

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
nx, ny = 80, 80
mesh = RectangleMesh(Point(0.0,0.0),Point(2.0,1.0),nx,ny)
boundaries = MeshFunction("size_t",mesh, mesh.topology().dim()-1)

#===================================;
#  Define function spaces for P1P1  ;
#===================================;
V = VectorFunctionSpace(mesh,"CG",1)
Q = FunctionSpace(mesh,"CG",1)
W = V * Q

# V = VectorFunctionSpace(mesh,"CG",2)
# Q = FunctionSpace(mesh,"DG",0)
# W = V * Q
#===========================;
#  Drag coefficient (mu/k)  ;
#===========================;
alpha = Constant(1.0)

#=====================;
#  Define boundaries  ;
#=====================;
class left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],0.0)

class right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near (x[0],2.0)

class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],0.0)

class topRigid(SubDomain):              
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],1.0) and \
            ( x[0] <= 0.95 or  x[0] >= 1.05) 

class topOutlet(SubDomain):              
    def inside(self, x, on_boundary):
       return on_boundary and near(x[1],1.0) and \
           ( x[0] > 0.95 and  x[0] < 1.05 ) 

right = right()
left = left()
bottom = bottom()
topRigid = topRigid()
topOutlet = topOutlet()

boundaries.set_all(0)
left.mark(boundaries,1)
right.mark(boundaries,2)
bottom.mark(boundaries,3)
topRigid.mark(boundaries,4)
topOutlet.mark(boundaries,5)

#=========================================;
# No-slip boundary condition for velocity ;
#=========================================;
bc_B = DirichletBC(W.sub(0).sub(1),Constant(0.0),bottom)
bc_T_rigid = DirichletBC(W.sub(0).sub(1),Constant(0.0),topRigid)

bcs = [bc_B,bc_T_rigid]

#=================;
# Pressure Values ;
#=================; 
inj_pressure = Constant(16000.0)
prod_pressure = Constant(0.0)

#==============================;
#  Expression for body force   ;
#==============================;
f = Constant((0.0,0.0))

#==============================;
#  Define variational problem  ;
#==============================;
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

ds = Measure("ds")[boundaries]
n = FacetNormal(mesh)

a = dot(alpha*u,v) * dx - div(v)*p*dx - q*div(u)*dx \
  - 0.5 * dot(alpha * v + grad(q),1.0/alpha * (alpha * u + grad(p))) * dx
L = inner(f,v) *dx - 0.5 * dot(alpha * v + grad(q),1.0/alpha * f) * dx \
  - dot(v,n) * inj_pressure * ds(1) \
  - dot(v,n) * inj_pressure * ds(2) \
  - dot(v,n) * prod_pressure * ds(5)

# a = dot(alpha*u,v) * dx - div(v)*p*dx - q*div(u)*dx
# L = inner(f,v) *dx \
#   - dot(v,n) * inj_pressure * ds(1) \
#   - dot(v,n) * inj_pressure * ds(2) \
#   - dot(v,n) * prod_pressure * ds(5)

#=================================;
#  Solve the resulting equations  ;
#=================================;
outfileV = File('Reservoir_Velocity.pvd')
outfileP = File('Reservoir_Pressure.pvd')

U = Function(W)
solve(a == L, U, bcs)

# Get sub-functions
u, p = U.split()

# Plot solution
outfileV << u
outfileP << p

#======;
# Flux ;
#======;
flux = assemble(dot(u, n)*ds(5))
print "Flux = ",flux

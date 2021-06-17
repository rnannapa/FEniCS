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
boundaries = FacetFunction("size_t",mesh, mesh.topology().dim()-1)
subdomain = CellFunction("size_t",mesh)

#===================================;
#  Define function spaces for P1P1  ;
#===================================;
V = VectorFunctionSpace(mesh,"CG",1)
Q = FunctionSpace(mesh,"CG",1)
W = V * Q

#===========================;
#  Drag coefficient (mu/k)  ;
#===========================;
alpha1 = Constant(1.0)
alpha2 = Constant(0.001)

#=====================;
#  Define boundaries  ;
#=====================;
class left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near (x[0],0.0)

class right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near (x[0],1.0)

class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near (x[1],0.0)

class top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near (x[1],1.0)

class Domain1(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[1], (0.0, 0.5)) and between(x[0], (0.0, 0.5)))

class Domain2(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[1], (0.0, 0.5)) and between(x[0], (0.5, 1.0)))

class Domain3(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[1], (0.5, 1.0)) and between(x[0], (0.0, 0.5)))

class Domain4(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[1], (0.5, 1.0)) and between(x[0], (0.5, 1.0)))

class injCorner(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] == 0.0 and x[1] == 0.0

class proCorner(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] == 1.0 and x[1] == 1.0


right   = right()
left    = left()
bottom  = bottom()
top     = top()
Domain1 = Domain1()
Domain2 = Domain2()
Domain3 = Domain3()
Domain4 = Domain4()
injCorner = injCorner()
proCorner = proCorner()

boundaries.set_all(0)
left.mark(boundaries,1)
right.mark(boundaries,2)
bottom.mark(boundaries,3)
top.mark(boundaries,4)

subdomain.set_all(0)
Domain1.mark(subdomain,1)
Domain2.mark(subdomain,2)
Domain3.mark(subdomain,3)
Domain4.mark(subdomain,4)


#=========================================;
# No-slip boundary condition for velocity ;
#=========================================;
bc_B = DirichletBC(W.sub(0).sub(1),Constant(0.0),bottom)
bc_T = DirichletBC(W.sub(0).sub(1),Constant(0.0),top)
bc_L = DirichletBC(W.sub(0).sub(0),Constant(0.0),left)
bc_R = DirichletBC(W.sub(0).sub(0),Constant(0.0),right)

bc_iC = DirichletBC(W.sub(1),Constant(+0.25),injCorner,method = "pointwise")
bc_pC = DirichletBC(W.sub(1),Constant(-0.25),proCorner,method = "pointwise")

bcs = [bc_B,bc_L,bc_R,bc_T,bc_iC,bc_pC]

#=================;
# Pressure Values ;
#=================; 
inj_pressure = Constant(100.0)
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
stab_param = 1e-3;
dx = Measure("dx")[subdomain]
ds = Measure("ds")[boundaries]

a = dot(alpha1*u,v) * dx(1) - div(v)*p*dx(1) - q*div(u)*dx(1) + stab_param * p * q * dx(1) +\
    dot(alpha2*u,v) * dx(2) - div(v)*p*dx(2) - q*div(u)*dx(2) + stab_param * p * q * dx(2) +\
    dot(alpha2*u,v) * dx(3) - div(v)*p*dx(3) - q*div(u)*dx(3) + stab_param * p * q * dx(3) +\
    dot(alpha1*u,v) * dx(4) - div(v)*p*dx(4) - q*div(u)*dx(4) + stab_param * p * q * dx(4)

L = inner(f,v)*dx(1) + inner(f,v)*dx(2) + inner(f,v)*dx(3) + inner(f,v)*dx(4)


# a = dot(alpha1*u,v) * dx(1) + dot(v,grad(p))*dx(1) - q*div(u)*dx(1) - \
# 	0.5 * dot(alpha1 * v + grad(q),1.0/alpha1 * (alpha1 * u + grad(p))) * dx(1)+\
# 	dot(alpha2*u,v) * dx(2) + dot(v,grad(p))*dx(2) - q*div(u)*dx(2) - \
# 	0.5 * dot(alpha2 * v + grad(q),1.0/alpha2 * (alpha2 * u + grad(p))) * dx(2)+\
# 	dot(alpha2*u,v) * dx(3) + dot(v,grad(p))*dx(3) - q*div(u)*dx(3) - \
# 	0.5 * dot(alpha2 * v + grad(q),1.0/alpha2 * (alpha2 * u + grad(p))) * dx(3)+\
#     dot(alpha1*u,v) * dx(4) + dot(v,grad(p))*dx(4) - q*div(u)*dx(4) - \
#     0.5 * dot(alpha1 * v + grad(q),1.0/alpha1 * (alpha1 * u + grad(p))) * dx(4)

# L = inner(f,v)*dx(1) - 0.5 * dot(alpha1 * v + grad(q),1.0/alpha1 * f) * dx(1) +\
# 	inner(f,v)*dx(2) - 0.5 * dot(alpha2 * v + grad(q),1.0/alpha2 * f) * dx(2) +\
# 	inner(f,v)*dx(3) - 0.5 * dot(alpha2 * v + grad(q),1.0/alpha2 * f) * dx(3) +\
# 	inner(f,v)*dx(4) - 0.5 * dot(alpha1 * v + grad(q),1.0/alpha1 * f) * dx(4)

#=================================;
#  Solve the resulting equations  ;
#=================================;
outfileV = File('Reservoir_Velocity.pvd')
outfileP = File('Reservoir_Pressure.pvd')

U = Function(W)
solve(a == L, U, bcs)

# Get sub-functions
u, p = U.split()
# u, p = split(U)

# Plot solution
# outfileV << u
# outfileP << p

#======;
# Flux ;
#======;
# flux = assemble(dot(u, n)*ds(5))
# print "Flux = ",flux

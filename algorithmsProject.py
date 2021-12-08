#https://www.realpythonproject.com/basic-linear-programming-in-python-with-pulp/
#https://nilearn.github.io/plotting/index.html#surface-plotting
#pip install pulp
from pulp import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

#verts = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]])
points = 3.0
theta = np.linspace(0,2.0-2.0/points,int(points))*3.14159
verts = np.zeros((int(points),3))
verts[:,0] = np.cos(theta)#array([ 1.00000000e+00,  1.32679490e-06, -1.00000000e+00, -3.98038469e-06])
verts[:,1] = np.sin(theta)#array([ 0.00000000e+00,  1.00000000e+00,  2.65358979e-06, -1.00000000e+00])
print(verts)
vert_to_cart = dict()
for i in range(verts.shape[0]):
    vert_to_cart['v'+str(i)] = (verts[i,0],verts[i,1],verts[i,2])
print(vert_to_cart)
verticies = list(vert_to_cart.keys())
print(verticies)
distances = distance.cdist(verts, verts, 'euclidean')
print(distances)
problem = LpProblem('Car Factory', LpMinimize)


edges = []

inci_dict = {i:[] for i in verticies}

edges_dict = dict()
cartesian_dict = dict()

for vi in range(len(verticies)):
    for vj in range(vi+1,len(verticies)):
        s = verticies[vi]+str('_')+verticies[vj]
        var = LpVariable(s, cat="Binary")
        var_edge_weights = (var,distances[vi,vj])
        var_degree = (var,1)
        inci_dict[verticies[vi]].append(var_degree)
        inci_dict[verticies[vj]].append(var_degree)
        edges.append(var_edge_weights)
        cartesian_dict[s]=(vert_to_cart[verticies[vi]][0],vert_to_cart[verticies[vi]][1],vert_to_cart[verticies[vj]][0],vert_to_cart[verticies[vj]][1])
        edges_dict[s] = var_edge_weights


assert len(edges)==(verts.shape[0]*(verts.shape[0]-1)//2)
obj = LpAffineExpression(edges)
print ('objective',obj)
problem+= obj

numedges = len(edges)


nVert = len(verticies)
assert numedges == nVert*(nVert-1)//2


for vert in verticies:
    assert len(inci_dict[verticies[vi]]) == verts.shape[0]-1
    print(inci_dict[verticies[vi]])
    #exit()
    ec = LpAffineExpression(inci_dict[vert])
    ec = LpConstraint(e=ec, sense=1, name='incivert_'+vert, rhs=2)
    problem+=ec


print("Current Status: ", LpStatus[problem.status])

problem.solve()
tot = 0.0
for edge in edges:
    tot+= edge[0].varValue
print("Total used edges: ", tot)

print(problem.objective)
for a,b in enumerate(problem.constraints):
    print(a,b)

fig = plt.figure()
ax = plt.axes(projection='3d')

for ind,key in enumerate(edges_dict):
    edge = edges_dict[key]
    print('edge[1],key,edge[0].varval:::',edge[1],key,edge[0].varValue)
    if edge[0].varValue == 1.0:
        x = [cartesian_dict[key][0],cartesian_dict[key][2]]
        y = [cartesian_dict[key][1],cartesian_dict[key][3]]
        plt.plot(x,y,color='green')
        
plt.show()

#https://www.realpythonproject.com/basic-linear-programming-in-python-with-pulp/
#https://nilearn.github.io/plotting/index.html#surface-plotting
from pulp import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import mutated_icospheres as mut

dataset = mut.MutatedIcospheresDataset(length=20)

mini_batch = 20


points = 3.0
import mutated_icospheres as mut
import random

n = [i for i in range(9002)]
inds = random.sample(n,100)

dataset = mut.MutatedIcospheresDataset(length=20)

mini_batch = 20

verts = dataset[0][1]
verts = verts[inds,:]
median = (np.median(verts[:,0]),np.median(verts[:,1]),np.median(verts[:,2]))


vert_to_cart = dict()
for i in range(verts.shape[0]):
    vert_to_cart['v'+str(i)] = (verts[i,0],verts[i,1],verts[i,2])
verticies = list(vert_to_cart.keys())
distances = distance.cdist(verts, verts, 'euclidean')
problem = LpProblem('Graph problem', LpMinimize)


edges = []

inci_dict = {i:[] for i in verticies}

edges_dict = dict()
cartesian_dict = dict()
count = 0

for vi in range(len(verticies)):
    for vj in range(vi+1,len(verticies)):
        count+=1
        s = verticies[vi]+str('_')+verticies[vj]
        var = LpVariable(s, cat="Binary")
        var_edge_weights = (var,distances[vi,vj])
        var_degree = (var,1)
        inci_dict[verticies[vi]].append(var_degree)
        inci_dict[verticies[vj]].append(var_degree)
        edges.append(var_edge_weights)
        cartesian_dict[s]=(vert_to_cart[verticies[vi]][0],vert_to_cart[verticies[vi]][1],vert_to_cart[verticies[vi]][2],vert_to_cart[verticies[vj]][0],vert_to_cart[verticies[vj]][1],vert_to_cart[verticies[vj]][2])
        edges_dict[s] = var_edge_weights


assert len(edges)==(verts.shape[0]*(verts.shape[0]-1)//2)
obj = LpAffineExpression(edges)
problem+= obj

numedges = len(edges)


nVert = len(verticies)
assert numedges == nVert*(nVert-1)//2


for vert in verticies:
    assert len(inci_dict[verticies[vi]]) == verts.shape[0]-1
    ec = LpAffineExpression(inci_dict[vert])
    ec = LpConstraint(e=ec, sense=1, name='incivert_'+vert, rhs=6)
    problem+=ec


print("Current Status: ", LpStatus[problem.status])

problem.solve()
tot = 0.0
for edge in edges:
    tot+= edge[0].varValue


fig = plt.figure()
ax = plt.axes(projection='3d')

#plot all
total_ILP_edges = 0

for ind,key in enumerate(edges_dict):
    edge = edges_dict[key]
    if edge[0].varValue == 1.0:
        total_ILP_edges += 1
        x = [cartesian_dict[key][0],cartesian_dict[key][3]]
        y = [cartesian_dict[key][1],cartesian_dict[key][4]]
        z = [cartesian_dict[key][2],cartesian_dict[key][5]]
        ax.plot(x,y,z,marker=',',color='green',lw=.5)

ax.scatter(verts[:,0],verts[:,1],verts[:,2],s=10)

plt.title('ILP implementation')

plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')

edges2 = []
rhs = 6
for i in range(distances.shape[0]):
    d = np.copy(distances[i])
    assert np.argmin(d) == i 
    maxd = np.max(d)
    d[i] = maxd
    for j in range(rhs):
        v = np.argmin(d)
        d[v] = maxd
        
        if i < v:
            edges2.append('v'+str(i)+'_v'+str(v))
        else:
            edges2.append('v'+str(v)+'_v'+str(i))

edges2 = set(edges2) #remove duplicates  
tot_greedy_d = 0.0
for key in edges2:
    edge = edges_dict[key]
    x = [cartesian_dict[key][0],cartesian_dict[key][3]]
    y = [cartesian_dict[key][1],cartesian_dict[key][4]]
    z = [cartesian_dict[key][2],cartesian_dict[key][5]]
    d = np.sqrt((x[0]-x[1])**2+(y[0]-y[1])**2+(z[0]-z[1])**2)
    ax.plot(x,y,z,marker=',',color='green',lw=.5)
    tot_greedy_d+=d

ax.scatter(verts[:,0],verts[:,1],verts[:,2],s=10)



print('tot_greedy_d',tot_greedy_d.item())
print('total greedy edges',len(edges2))
print('objective/total distance ilp',float(pulp.value(obj)))
print('total_ILP_edges',total_ILP_edges)
plt.title('greedy implementation')
plt.show()

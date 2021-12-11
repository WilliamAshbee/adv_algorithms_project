#https://www.realpythonproject.com/basic-linear-programming-in-python-with-pulp/
#https://nilearn.github.io/plotting/index.html#surface-plotting
#pip install pulp
from pulp import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import mutated_icospheres as mut

dataset = mut.MutatedIcospheresDataset(length=20)

mini_batch = 20

print(dataset[0][1].shape)

#verts = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]])
points = 3.0
#theta = np.linspace(0,2.0-2.0/points,int(points))*3.14159
#verts = np.zeros((int(points),3))
#verts[:,0] = np.cos(theta)#array([ 1.00000000e+00,  1.32679490e-06, -1.00000000e+00, -3.98038469e-06])
#verts[:,1] = np.sin(theta)#array([ 0.00000000e+00,  1.00000000e+00,  2.65358979e-06, -1.00000000e+00])
import mutated_icospheres as mut
import random

n = [i for i in range(9002)]
inds = random.sample(n,100)

dataset = mut.MutatedIcospheresDataset(length=20)

mini_batch = 20

print(dataset[0][1].shape)
verts = dataset[0][1]

#exit()
print(verts.shape)
verts = verts[inds,:]
median = (np.median(verts[:,0]),np.median(verts[:,1]),np.median(verts[:,2]))

print(verts.shape,type(verts))
print(verts.shape[0],type(verts.shape[0]))

vert_to_cart = dict()
for i in range(verts.shape[0]):
    vert_to_cart['v'+str(i)] = (verts[i,0],verts[i,1],verts[i,2])
print(len(vert_to_cart))
verticies = list(vert_to_cart.keys())
print(len(verticies))
distances = distance.cdist(verts, verts, 'euclidean')
print(distances)
problem = LpProblem('Graph problem', LpMinimize)


edges = []

inci_dict = {i:[] for i in verticies}

edges_dict = dict()
cartesian_dict = dict()
count = 0

for vi in range(len(verticies)):
    for vj in range(vi+1,len(verticies)):
        if count % 1000 ==0:
            print('count',count)
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
#print ('objective',obj)
problem+= obj

numedges = len(edges)


nVert = len(verticies)
assert numedges == nVert*(nVert-1)//2


for vert in verticies:
    assert len(inci_dict[verticies[vi]]) == verts.shape[0]-1
    #print(inci_dict[verticies[vi]])
    #exit()
    ec = LpAffineExpression(inci_dict[vert])
    ec = LpConstraint(e=ec, sense=1, name='incivert_'+vert, rhs=6)
    problem+=ec


print("Current Status: ", LpStatus[problem.status])

problem.solve()
tot = 0.0
for edge in edges:
    tot+= edge[0].varValue
print("Total used edges: ", tot)

#print(problem.objective)
# for a,b in enumerate(problem.constraints):
#     print(a,b)

fig = plt.figure()
ax = plt.axes(projection='3d')

omitset = set()


#plot half
for ind,key in enumerate(edges_dict):
    print(ind,key)
    edge = edges_dict[key]
    #print('edge[1],key,edge[0].varval:::',edge[1],key,edge[0].varValue)
    if edge[0].varValue == 1.0  :
        x = [cartesian_dict[key][0],cartesian_dict[key][3]]
        y = [cartesian_dict[key][1],cartesian_dict[key][4]]
        z = [cartesian_dict[key][2],cartesian_dict[key][5]]
        if x[0] < median[0] and x[1] < median[0]:
            ax.plot(x,y,z,marker=',',color='green',lw=.5)

ax.scatter(verts[verts[:,0]<median[0],0],verts[verts[:,0]<median[0],1],verts[verts[:,0]<median[0],2])


#ax.view_init(0,0,0)
#plt.savefig('first.png',dpi=600)
plt.show()

#plot all
for ind,key in enumerate(edges_dict):
    print(ind,key)
    edge = edges_dict[key]
    #print('edge[1],key,edge[0].varval:::',edge[1],key,edge[0].varValue)
    if edge[0].varValue == 1.0  :
        x = [cartesian_dict[key][0],cartesian_dict[key][3]]
        y = [cartesian_dict[key][1],cartesian_dict[key][4]]
        z = [cartesian_dict[key][2],cartesian_dict[key][5]]
        ax.plot(x,y,z,marker=',',color='green',lw=.5)

ax.scatter(verts[:,0],verts[:,1],verts[:,2])


#ax.view_init(0,0,0)
#plt.savefig('first.png',dpi=600)
plt.show()

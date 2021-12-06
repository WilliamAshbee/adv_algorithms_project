#https://www.realpythonproject.com/basic-linear-programming-in-python-with-pulp/
#https://nilearn.github.io/plotting/index.html#surface-plotting
#pip install pulp
from pulp import *
import numpy as np
import matplotlib.pyplot as plt

vert_to_cart = {'a':(0,0),'b':(1,0),'c':(1,1),'d':(0,1)}
verticies = ['a','b','c','d']
distances = np.array([[0,1,np.sqrt(2),1],[1,0,1,np.sqrt(2)],[np.sqrt(2),1,0,1],[1,np.sqrt(2),1,0]])
problem = LpProblem('Car Factory', LpMinimize)


edges = []

inci_dict = {i:[] for i in verticies}

edges_dict = dict()
cartesian_dict = dict()

for vi in range(len(verticies)):
    for vj in range(vi+1,len(verticies)):
        s = verticies[vi]+str('_')+verticies[vj]
        var = (LpVariable(s, cat="Binary"),distances[vi,vj])
        inci_dict[verticies[vi]].append(var)
        inci_dict[verticies[vj]].append(var)
        edges.append(var)
        cartesian_dict[s]=(vert_to_cart[verticies[vi]][0],vert_to_cart[verticies[vi]][1],vert_to_cart[verticies[vj]][0],vert_to_cart[verticies[vj]][1])
        edges_dict[s] = var
assert len(edges)==6
obj = LpAffineExpression(edges)
print ('objective',obj)
problem+= obj

numedges = len(edges)


nVert = len(verticies)
assert numedges == nVert*(nVert-1)//2


for vert in verticies:
    assert len(inci_dict[verticies[vi]]) == 3
    ec = LpAffineExpression(inci_dict[vert])
    ec = LpConstraint(e=ec, sense=1, name='incivert_'+vert, rhs=2)
    problem+=ec


print("Current Status: ", LpStatus[problem.status])

problem.solve()

for edge in edges:
    print('edge',edge[1],edge[0].varValue)


print("Total edges: ", value(problem.objective))

print(problem.objective)
for a,b in enumerate(problem.constraints):
    print(a,b)

fig = plt.figure()
ax = plt.axes(projection='3d')

for ind,key in enumerate(edges_dict):
    edge = edges_dict[key]
    print(edge,key)
    if edge[0].varValue == 1.0:
        x = [cartesian_dict[key][0],cartesian_dict[key][2]]
        y = [cartesian_dict[key][1],cartesian_dict[key][3]]
        plt.plot(x,y,color='green')
        
plt.show()
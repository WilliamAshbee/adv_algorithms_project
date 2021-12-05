#https://www.realpythonproject.com/basic-linear-programming-in-python-with-pulp/
#https://nilearn.github.io/plotting/index.html#surface-plotting
#pip install pulp
from pulp import *
import numpy as np
#  
#      a-b
#      | |
#      c-d
#

verticies = ['a','b','c','d']
distances = np.array([[0,1,1,1.5],[1,0,1.5,1],[1,1.5,0,1],[1.5,1,1,0]])
problem = LpProblem('Car Factory', LpMinimize)


edges = []

inci_dict = {i:[] for i in verticies}

for vi in range(len(verticies)):
    for vj in range(vi+1,len(verticies)):
        s = verticies[vi]+str('_')+verticies[vj]
        var = (LpVariable(s, cat="Binary"),distances[vi,vj])
        inci_dict[verticies[vi]].append(var)
        inci_dict[verticies[vj]].append(var)
        
        
        edges.append(var)

assert len(edges)==6
# # # ab = LpVariable('ab', cat="Binary")
# # # ac = LpVariable('ac', cat="Binary")
# # # ad = LpVariable('ad', cat="Binary")
# # # bc = LpVariable('bc', cat="Binary")
# # # bd = LpVariable('bd', cat="Binary")
# # # cd = LpVariable('cd', cat="Binary")
# # # edges = [(ab,1),(ac,1),(ad,1.5),(bc,1.5),(bd,1),(cd,1)]

#Objective Function
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

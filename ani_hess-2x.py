# calculate energy and gradient with ANI
import torch
import torchani
import math
import numpy as np

import sys
device = torch.device('cpu')

inp_f = "mol.tmp"



with open(inp_f,"r") as f:
    natom = int(f.readline())
    l1 = []
    l3 = []
    for i in range(natom):
       #l1.append(
        l0 = f.readline().split()
        l2 = l0[1:4]  
        x = float(l2[0]) * 0.529177
        y = float(l2[1]) * 0.529177
        z = float(l2[2]) * 0.529177
        l1.append([x,y,z])
        l3.append(int(l0[0]))
 



model = torchani.models.ANI2x(periodic_table_index=True).to(device).double()

species = torch.tensor(np.array(l3), device=device, dtype=torch.long).unsqueeze(0)


coordinates = torch.from_numpy(np.array(l1)).unsqueeze(0).requires_grad_(True)


masses = torchani.utils.get_atomic_masses(species)
energies = model((species, coordinates)).energies




# print polarizability and dipole derivatives
print("%20.12F%20.12F%20.12F"% (0,0,0))
print("%20.12F%20.12F%20.12F"% (0,0,0))
for i in range(3*natom):
  print("%20.12F%20.12F%20.12F"% (0,0,0))


hessian = torchani.utils.hessian(coordinates, energies=energies)
F1_mat = hessian.numpy()[0] * 0.529177 * 0.529177
F1_lt = F1_mat[np.tril_indices(3*natom)]
k = int(len(F1_lt)/3)
for i in range(k):
   a = F1_lt[i*3]
   b = F1_lt[i*3+1]
   c = F1_lt[i*3+2]
   print( "%20.12F%20.12F%20.12F"% tuple([a,b,c]) )


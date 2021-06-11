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

E1 = energies.item()

print("%20.12F%20.12F%20.12F%20.12F"% (E1,0,0,0))

derivative = torch.autograd.grad(energies.sum(), coordinates)[0]

G1 = derivative.numpy()[0] * 0.529177
for i in G1:
  print("%20.12F%20.12F%20.12F"% tuple(i))


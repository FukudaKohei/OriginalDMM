import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import os

DMM_dics = torch.load(os.path.join("saveData", "20211215_21_18","DMM_dic"))
pos = DMM_dics["mini_batch"][0]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.rcParams["font.size"] = 10
plt.plot(pos[:,0], pos[:,1], pos[:,2] , label="Training data")
plt.title("Lorentz Curves")
plt.legend()
fig.savefig("Lorentz"+".png")
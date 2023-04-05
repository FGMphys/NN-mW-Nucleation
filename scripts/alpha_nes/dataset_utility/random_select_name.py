import glob as gl
import numpy as np
import sys

def sort_fun(name):
    return int(name.split(sys.argv[1])[1])

pos_name=gl.glob(sys.argv[1]+'*')
if len(pos_name)<1:
   print(pos_name,len(pos_name))
   sys.exit("not_found")
pos_name.sort(key=sort_fun)

dim=len(pos_name)
number_of_frames=int(sys.argv[2])
vec=np.arange(0,dim)
seed=int(sys.argv[3])
np.random.seed(seed=seed)
for k in range(100):
    out=np.random.shuffle(vec)
    np.random.seed(seed=seed)

for el in vec[:number_of_frames]:
    print(pos_name[el],end=' ')

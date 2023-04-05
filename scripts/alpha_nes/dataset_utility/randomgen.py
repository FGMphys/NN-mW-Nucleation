import numpy as np
import sys
import os

if len(sys.argv)!=5:
   sys.exit("python "+sys.argv[0]+" [number of frame to select] [dt saving time step] [start time] [end time]")
nf=int(sys.argv[1])
dt=int(sys.argv[2])
start=int(sys.argv[3])
end=int(sys.argv[4])

np.random.seed(12345)
vec=np.arange(start,end,dt)
vec=np.random.choice(vec,nf,replace=False)

namefolder="random-pick-"+str(nf)+"fr"
os.mkdir(namefolder)
for el in vec:
    os.system("cp pos_"+str(el)+" "+str(namefolder))
    os.system("cp force_"+str(el)+" "+str(namefolder))




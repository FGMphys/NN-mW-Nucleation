import numpy as np
import sys



data=np.loadtxt(sys.argv[1])
val=float(sys.argv[2])
for k in range(data.shape[0]):
    if data[k,0]==val:
       found=data[k,1]
       break
try:   
   print(found)
except:
   print("No value found")

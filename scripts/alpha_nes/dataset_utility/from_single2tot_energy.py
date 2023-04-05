import numpy as np
import sys 

data=np.loadtxt(sys.argv[1],comments='"')
data=np.reshape(data,(-1,int(sys.argv[2])))
data=np.mean(data,axis=1)
np.savetxt(sys.argv[1],data,header='\"Energy\"',comments='')


#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

if len(sys.argv)!=4:
   sys.exit(sys.argv[0]+"[num particles] [raw energy path] [raw force path]")

print("maketraintest: Program is agnostic on which units the user chooses, but energy are expected to be normalizied for the total number of particles yet.")

os.mkdir("dataset")
os.mkdir("dataset/training")
os.mkdir("dataset/test")
seed_shuffle=12345

N=int(sys.argv[1])

el='descriptor_dataset.csv'
dataset=pd.read_csv(el,dtype='float64')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,ndes))
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

nf_tr=np.shape(data_tr)[0]
nf_ts=np.shape(data_ts)[0]

np.save("dataset/training/descriptors.npy",data_tr)
np.save("dataset/test/descriptors.npy",data_ts)
del dataset
del data_tr
del data_ts

el='descriptor_3bsupp_dataset.csv'
dataset=pd.read_csv(el,dtype='float64')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,ndes))
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

nf_tr=np.shape(data_tr)[0]
nf_ts=np.shape(data_ts)[0]

np.save("dataset/training/des3bsupp.npy",data_tr)
np.save("dataset/test/des3bsupp.npy",data_ts)
del dataset
del data_tr
del data_ts








el='radialderivat_dataset.csv'
dataset=pd.read_csv(el,dtype='float64')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,3,ndes))
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

np.save("dataset/training/intder2b.npy",data_tr)
np.save("dataset/test/intder2b.npy",data_ts)
del dataset
del data_tr
del data_ts

el='3bsupp_derivat_dataset.csv'
dataset=pd.read_csv(el,dtype='float64')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,3,ndes))
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

np.save("dataset/training/intder3bsupp.npy",data_tr)
np.save("dataset/test/intder3bsupp.npy",data_ts)
del dataset
del data_tr
del data_ts




el='radialint_dataset.csv'
dataset=pd.read_csv(el,dtype='int32')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,ndes))
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

np.save("dataset/training/int2b.npy",data_tr)
np.save("dataset/test/int2b.npy",data_ts)
del dataset
del data_tr
del data_ts


el='tripletsint_dataset.csv'
dataset=pd.read_csv(el,dtype='int32')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,ndes))
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

np.save("dataset/training/int3b.npy",data_tr)
np.save("dataset/test/int3b.npy",data_ts)
del dataset
del data_tr
del data_ts

el='tripletsderivat_dataset.csv'
dataset=pd.read_csv(el,dtype='float64')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,3,ndes))
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

np.save("dataset/training/intder3b.npy",data_tr)
np.save("dataset/test/intder3b.npy",data_ts)
del dataset
del data_tr
del data_ts

try:
    el=sys.argv[2]
    dataset=pd.read_csv(el,dtype='float64')
    dataset = dataset.iloc[:].values
    ndes=dataset.shape[1]
    dataset=np.reshape(dataset,(-1))
    np.random.seed(1)
    np.random.shuffle(dataset)
    data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

    np.save("dataset/training/energy.npy",data_tr)
    np.save("dataset/test/energy.npy",data_ts)
    del dataset
    del data_tr
    del data_ts
except:
    print("No dataset on energies has been found. I expect you want to run a force-matching procedure. If not check the energy path")
    np.save("dataset/training/energy.npy",np.zeros((nf_tr)))
    np.save("dataset/test/energy.npy",np.zeros((nf_ts)))
el=sys.argv[3]
dataset=pd.read_csv(el,dtype='float64',sep=' ')
dataset = dataset.iloc[:,:].values
dataset=np.reshape(dataset,(-1,N,3))
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

np.save("dataset/training/force.npy",data_tr)
np.save("dataset/test/force.npy",data_ts)
del dataset
del data_tr
del data_ts
try:
    el='dataset_position.dat'
    dataset=np.loadtxt(el,dtype='float64')
    dataset=dataset.reshape((-1,5+3*N))
    np.random.seed(1)
    np.random.shuffle(dataset)
    data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

    np.save("dataset/training/position.npy",data_tr)
    np.save("dataset/test/position.npy",data_ts)
    del dataset
    del data_ts
    del data_tr
except:
    print("make_train_test: no poistion found!")

try:
    el='dataset_pressure.dat'
    dataset=pd.read_csv(el,dtype='float64')
    dataset = dataset.iloc[:].values
    ndes=dataset.shape[1]
    dataset=np.reshape(dataset,(-1))
    np.random.seed(1)
    np.random.shuffle(dataset)
    data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)

    np.save("dataset/training/pressure.npy",data_tr)
    np.save("dataset/test/pressure.npy",data_ts)
    del dataset
    del data_tr
    del data_ts
except:
    print("make_train_test: no pressure data found!")

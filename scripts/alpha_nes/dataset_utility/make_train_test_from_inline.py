#!/usr/bin/env python3
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf

root_path='/home/francegm/programmi/alpha_nes_local'
descriptor_sopath=root_path+'/descriptors_utility/distangoli_py/op_comp_3bcc.so'


make_descriptor=tf.load_op_library(descriptor_sopath)

def checkdim_sca_vec(sca,vec):
    if len(vec)!=sca:
       sys.exit("alpha_nes: vector are too large or too small!")
    return 0

def make_typemap(tipos):
    num=0
    list_tmap=[]
    for el in tipos:
        for k in range(el):
            list_tmap.append(num)
        num=num+1
    return list_tmap
def read_data():
    pos=np.loadtxt("pos.dat")
    box=np.loadtxt("box.dat")
    energy=np.loadtxt("energy.dat")
    force=np.loadtxt("force.dat")
    tipos=np.loadtxt("type.dat")
    return pos,box,energy,force,tipos
def check_dimension(lista,axis):
    dim_prev=lista[0].shape[axis]
    for el in lista:
        dim=el.shape[axis]
        if dim!=dim_prev:
           sys.exit("Make_train_test_from_inline: error on dataset dimension along axis ",axis)
    return 1
def check_type_map(N,tipos):
    if N!=np.sum(tipos):
       sys.exit("Make_train_test_from_inline: Number of particle in type.dat file does not match number of particle in the other dataset!")
    return 1
def read_cutoff_info(args):
    rc=float(args[1])
    rad_buff=int(args[2])
    rc_ang=float(args[3])
    ang_buff=int(args[4])
    return rc,rad_buff,rc_ang,ang_buff
def compute_descriptors(rc,rad_buff,rc_ang,ang_buff,N,pos,boxdim,box,nf):
    fr=0
    res=make_descriptor.compute_descriptors(rc,rad_buff,rc_ang,ang_buff,N,pos[fr],boxdim,box[fr],1)
    for fr in range(1,nf):
        #res=[des,des3bsupp,intmap2b,intmap3b,der2b,der3b,der3bsupp]
        res_actual=make_descriptor.compute_descriptors(rc,rad_buff,rc_ang,ang_buff,N,pos[fr],boxdim,box[fr],1)
        res=[tf.concat([res[el],res_actual[el]],axis=0) for el in range(7)]

    return res
######################################################

if len(sys.argv)!=5:
   sys.exit(sys.argv[0]+"[two body cut-off] [two-body buffer] [three body cut-off] [three-body buffer]")

print("Make_train_test_from_inline: Program is agnostic on which units the user chooses. Energies must be total potential energies.")


[pos,box,energy,force,tipos]=read_data()
nf=int(pos.shape[0])
N=int(pos.shape[1]/3)
check_dimension([pos,force,box,energy],axis=0)
check_dimension([pos,force],axis=1)
check_type_map(N,tipos)

[rc,rad_buff,rc_ang,ang_buff]=read_cutoff_info(sys.argv)
print("Make_train_test_from_inline: Rc ",rc," radial buffer ",rad_buff," Rc_ang ",rc_ang," angular buffer ",ang_buff)


#[des,des3bsupp,intmap2b,intmap3b,der2b,der3b,der3bsupp]
descriptor_dataset=compute_descriptors(rc,rad_buff,rc_ang,ang_buff,N,pos,6,box,nf)
label_des=['descriptors.npy','des3bsupp.npy','int2b.npy','int3b.npy',
           'intder2b.npy','intder3b.npy','intder3bsupp.npy']

#####Make dataset folder #####
os.mkdir("dataset")
os.mkdir("dataset/training")
os.mkdir("dataset/test")
seed_shuffle=12345
np.savetxt("dataset/type.dat",tipos.reshape(-1,1),fmt='%d')
for k,el in enumerate(descriptor_dataset):
    dataset=el.numpy()
    np.random.seed(1)
    np.random.shuffle(dataset)
    data_tr,data_ts = train_test_split(dataset,train_size=0.8,
                                       random_state=seed_shuffle,
                                       shuffle=True)
    np.save("dataset/training/"+label_des[k],data_tr)
    np.save("dataset/test/"+label_des[k],data_ts)

dataset=np.reshape(energy,(-1))
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)
np.save("dataset/training/energy.npy",data_tr)
np.save("dataset/test/energy.npy",data_ts)

dataset=force.reshape(nf,N,3)
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)
np.save("dataset/training/force.npy",data_tr)
np.save("dataset/test/force.npy",data_ts)



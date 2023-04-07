#IMPORT I/O file
import os
import time
import sys
import tensorflow as tf

from numpy.random import seed

from numpy.random import default_rng


import numpy as np
import yaml

from gradient_utility import gradforceop_triplNODEN
from gradient_utility import gradforceop_rNODEN
from gradient_utility import grad_twobody_par
from gradient_utility import grad_threebody_par

from full_model_alphanes.alpha_nes_full_mod_inference import alpha_nes_full_inference

from source_routine.physics_layer_mod import physics_layer
from source_routine.physics_layer_mod import lognorm_layer

from source_routine.force_layer_mod import force_layer

@tf.function()
def MSE(ypred,y):
   loss_function=tf.reduce_mean(tf.square((ypred-y)))
   return loss_function
def make_dataset_stream(base_pattern,mode):
    energy_on_disk=np.load(base_pattern+'/'+mode+'/'+'energy.npy',mmap_mode='r')
    force_on_disk=np.load(base_pattern+'/'+mode+'/'+'force.npy',mmap_mode='r')

    descri_on_disk=np.load(base_pattern+'/'+mode+'/'+'descriptors.npy',mmap_mode='r')
    des3bsupp_on_disk=np.load(base_pattern+'/'+mode+'/'+'des3bsupp.npy',mmap_mode='r')

    int2b_on_disk=np.load(base_pattern+'/'+mode+'/'+'int2b.npy',mmap_mode='r')
    int3b_on_disk=np.load(base_pattern+'/'+mode+'/'+'int3b.npy',mmap_mode='r')

    intder2b_on_disk=np.load(base_pattern+'/'+mode+'/'+'intder2b.npy',mmap_mode='r')
    intder3b_on_disk=np.load(base_pattern+'/'+mode+'/'+'intder3b.npy',mmap_mode='r')
    intder3bsupp_on_disk=np.load(base_pattern+'/'+mode+'/'+'intder3bsupp.npy',mmap_mode='r')


    return energy_on_disk,force_on_disk,descri_on_disk,des3bsupp_on_disk,int2b_on_disk,int3b_on_disk,intder2b_on_disk,intder3b_on_disk,intder3bsupp_on_disk


def check_dimension(buffdim,dimension,mode):
    res=buffdim
    if buffdim>dimension:
       print("alpha_nes: buffdim in ",mode," mode is bigger than number of frames in the dataset. We set buddim=datasetdim!")
       res=dimension
    return res

def make_idx_str(dimension,buffdim,mode):
    buffdim=check_dimension(buffdim,dimension,mode)
    truedim=dimension//buffdim*buffdim
    rejected=dimension%buffdim
    print("\nalpha_nes: It will be rejected ",rejected,' frames picked randomly to ensure batch size and buffer requested.\n')
    vec=np.arange(0,dimension)
    np.random.shuffle(vec)
    vec=np.reshape(vec[:truedim],(dimension//buffdim,buffdim))
    return buffdim,vec

def check_along_frames(list_of_arr,axis):
    ref=list_of_arr[0].shape[axis]
    for el in list_of_arr:
        if ref!=el.shape[axis]:
           sys.exit("Dataset are not valid. Error on dimension along axis "+str(axis))
    return 0

def save_info(quant,out_folder,numel):
    k=0
    for idqu,qu in enumerate(quant):
        for idel,el in enumerate(qu):
            if el.numpy().shape==():
               np.savetxt(out_folder+'/'+str(idqu)+"_quant_"+str(numel+idel),el.numpy().reshape((1,1)))
            else:
               np.savetxt(out_folder+'/'+str(idqu)+"_quant_"+str(numel+idel),el.numpy())
            k=k+1
    return k+numel





################# MAIN #########################################################
##Read the input file
with open(sys.argv[1]) as file:
    full_param = yaml.load(file, Loader=yaml.FullLoader)

#Read dataset map on disk
base_pattern=full_param['dataset_folder']
[e_map_ts,f_map_ts,des_map_ts,des3bsupp_map_ts,int2b_map_ts,int3b_map_ts,intder2b_map_ts,
intder3b_map_ts,intder3bsupp_map_ts]=make_dataset_stream(base_pattern,'test')

check_along_frames([des_map_ts,int2b_map_ts,int3b_map_ts,intder2b_map_ts,intder3b_map_ts],0)
#Building a stream vector
buffer_stream_ts=full_param['buffer_stream_dim_ts']
dimts=des_map_ts.shape[0]
N=des_map_ts.shape[1]

[buffer_stream_ts,idx_str_ts]=make_idx_str(dimts,buffer_stream_ts,'test')

###Initialize the Encoder and Decoder
tf.keras.backend.set_floatx('float64')

#Open the saved model
modelname=full_param['model_name']


init_alpha2b=np.loadtxt(modelname+"/alpha_2body.dat")
init_alpha3b=np.loadtxt(modelname+"/alpha_3body.dat")
nalpha_r=np.shape(init_alpha2b)[0]
nalpha_a=np.shape(init_alpha3b)[0]


initial_type_emb=[np.ones(1),np.ones(1)]

Physics_Layer=physics_layer(init_alpha2b,init_alpha3b,initial_type_emb)

init_mu=np.loadtxt(modelname+"/alpha_mu.dat")
Lognorm_Layer=lognorm_layer(nalpha_r+nalpha_a,init_mu)



Force_Layer=force_layer()



tipos=[1000]
type_map=[0 for k in range(N)]
model=alpha_nes_full_inference(Physics_Layer,Force_Layer,Lognorm_Layer,
                               tipos,type_map,modelname)
testmeth=model.full_test

delta_e=0.
delta_f=0.
count=0.
for stepts,el in enumerate(idx_str_ts):
    datasetcomplete = tf.data.Dataset.from_tensor_slices((des_map_ts[el],des3bsupp_map_ts[el],
                                                          int2b_map_ts[el],intder2b_map_ts[el],
                                                          int3b_map_ts[el],intder3b_map_ts[el],
                                                          intder3bsupp_map_ts[el],e_map_ts[el],
                                                          f_map_ts[el])).batch(buffer_stream_ts)
    for step2, (x,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp,ene_true,force_true) in enumerate(datasetcomplete):
        [energy,force]=testmeth(x,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp)
        delta_e+=MSE(energy,ene_true).numpy()
        delta_f+=MSE(force,force_true).numpy()
        count=count+1
print("\n RMSE_Energy    RMSE_Force")
print(np.sqrt(delta_e/count),np.sqrt(delta_f/count),sep='   ')

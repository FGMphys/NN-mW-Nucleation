import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

root_path='/home/francegm/programmi/alpha_nes'
compute_2b_pargrad = tf.load_op_library(root_path+'/bin/alphagrad_2body.so')

@ops.RegisterGradient("ComputeSortProj")
def _compute_sort_proj_grad(op,grad):
    [net_grad,grad_emb2b_afs] =  compute_2b_pargrad.compute_two_body_par_grad (
                                                 grad,op.inputs[0],
                                                 op.inputs[1],op.inputs[2],
                                                 op.inputs[3],op.inputs[4],
                                                 op.inputs[5],op.inputs[6],
                                                 op.inputs[7],op.inputs[8],
                                                 op.inputs[9])
    return [None, None, None,None,None,net_grad,None,grad_emb2b_afs,None,None]

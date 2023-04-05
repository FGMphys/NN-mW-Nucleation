import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

root_path='/home/francegm/programmi/alpha_nes'
compforcegradtripl_module = tf.load_op_library(root_path+'/bin/computeforcegrad_triplNODEN_alpha2.so')

@ops.RegisterGradient("ComputeForceTripl")
def _compute_force_tripl_grad(op, grad):
    [net_grad0,net_grad1,grad_emb3b_par] =  compforcegradtripl_module.compute_force_tripl_grad (grad,
                                                 op.inputs[0],op.inputs[1],
                                                 op.inputs[2],op.inputs[3],
                                                 op.inputs[4],op.inputs[5],
                                                 op.inputs[6],op.inputs[7],
                                                 op.inputs[8],op.inputs[9],
                                                 op.inputs[10],op.inputs[11],
                                                 op.inputs[12],op.inputs[13],
                                                 op.inputs[14],op.inputs[15])
    return [net_grad0,None,None,None,None,None,None,None,None,None,None,None,net_grad1,grad_emb3b_par,None,None]

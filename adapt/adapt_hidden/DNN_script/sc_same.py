import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import logging
import time
import os
from config_util import parse_args, get_checkpoint_path, parse_contexts

args = parse_args()
args.config.write(sys.stdout)
device = mx.gpu(0)
sc_prefix = args.config.get('adapt_sc', 'sc_prefix')
B_model = mx.model.FeedForward.load(sc_prefix, 100)

B_arg_params=B_model.arg_params

W_names = ['fc1_weight','fc2_weight','fc3_weight','fc4_weight']
B_names = ['B1_bias','B2_bias','B3_bias','B4_bias']


load_prefix = args.config.get('adapt_sc', 'load_prefix')
V_model = mx.model.FeedForward.load(load_prefix, 30)

V_arg_params=V_model.arg_params
V_names = ['sc1_weight','sc2_weight','sc3_weight','sc4_weight']
b_names = ['fc1_bias','fc2_bias','fc3_bias','fc4_bias']


Vsc1 = B_arg_params['B1_bias'].asnumpy() - V_arg_params['fc1_bias'].asnumpy() 
Vsc2 = B_arg_params['B2_bias'].asnumpy()  - V_arg_params['fc2_bias'].asnumpy() 
Vsc3= B_arg_params['B3_bias'].asnumpy() - V_arg_params['fc3_bias'].asnumpy() 
Vsc4= B_arg_params['B4_bias'].asnumpy()  - V_arg_params['fc4_bias'].asnumpy() 
print Vsc1.shape
Vsc=np.vstack((Vsc1, Vsc2, Vsc3, Vsc4))

V=np.vstack((V_arg_params['sc1_weight'].asnumpy(), V_arg_params['sc2_weight'].asnumpy(), V_arg_params['sc3_weight'].asnumpy(), V_arg_params['sc4_weight'].asnumpy()))

np.savetxt('Vsc.txt', Vsc)
np.savetxt('V.txt', V)
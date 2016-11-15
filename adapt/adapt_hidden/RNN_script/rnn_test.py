import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import logging
import time
import os
from config_util import parse_args, get_checkpoint_path, parse_contexts
import scipy.io as scio
from Error import get_err
from hybrid import hybrid_net, lstm_unroll, LSTMInferenceModel
from DNN_data_read import read_data_train, my_DataRead

from io_util import TruncatedSentenceIter
from load_data import my_DataRead

if __name__ == '__main__':
	args = parse_args()
	args.config.write(sys.stdout)

	feat_dim = args.config.getint('data', 'xdim')
	label_dim = args.config.getint('data', 'ydim')

	test_input = args.config.get('test', 'test_input')
	test_target = args.config.get('test', 'test_target')
	xdim = args.config.getint('data', 'xdim')
	ydim = args.config.getint('data', 'ydim')

	test_prefix = args.config.get('test', 'test_prefix')
	load_epoch_num = args.config.getint('test', 'load_epoch_num')

	out_file = args.config.get('test', 'out_file')
	isExists=os.path.exists(out_file)
	if not isExists:
		 os.makedirs(out_file)

	contexts = parse_contexts(args)

	test_data_args = {
			"gpu_chunk": 32768,
			"input_file": test_input,
			"target_file": test_target,
			"xdim": xdim,
			"ydim": ydim
			}
	test_sets = [my_DataRead(test_data_args,20)]

	batch_size = args.config.getint('train', 'batch_size')
	truncate_len = args.config.getint('train', 'truncate_len')
	num_hidden_lstm = args.config.getint('arch', 'num_hidden_lstm')
	num_lstm_layer = args.config.getint('arch', 'num_lstm_layer')

	init_c = [('l%d_init_c'%l, (batch_size, num_hidden_lstm)) for l in range(num_lstm_layer)]
	
	init_h = [('l%d_init_h'%l, (batch_size, num_hidden_lstm)) for l in range(num_lstm_layer)]

	init_states = init_c + init_h

	state_names = [x[0] for x in init_states]

	data_names = [data_name] + state_names

	data_test = TruncatedSentenceIter(test_sets,batch_size, init_states,truncate_len=truncate_len, delay=10, feat_dim=feat_dim, label_dim=label_dim, data_name='data', label_name='linear_label', do_shuffling=False, pad_zeros=False)

	data_names = [x[0] for x in data_test.provide_data]
	label_names = [x[0] for x in data_test.provide_label]



	sym, arg_params, aux_params = mx.model.load_checkpoint(test_prefix, load_epoch_num)
	mod = mx.mod.Module(sym, context=contexts, data_names=data_names,label_names=label_names)
	mod.bind(data_shapes=data_test.provide_data, label_shapes=data_test.provide_label, for_training=False)
	mod.set_params(arg_params, aux_params)
	outputs	=mod.predict(data_test)

	print outputs.shape

	max_feat = np.loadtxt("max_feat.txt")
	max_label = np.loadtxt("max_label.txt")
	min_label = np.loadtxt("min_label.txt")
	result = np.multiply(outputs, (max_label-min_label))+min_label



	print result

	(test_feats, test_tgts, test_ids, test_fr) = read_data_train(test_sets[0])

	(mcd,rmse,bap_err,vu_err) = get_err(test_tgts,result)
	f = open(out_file+"/err.txt", 'w')
	f.write( "mcd: %f \n" % mcd)
	f.write( "rmse: %f \n" % rmse)
	f.write( "bap_err: %f \n" % bap_err)
	f.write("vu: %f \n" % vu_err)
	f.close()



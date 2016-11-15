import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import logging
import time
import os
from config_util import parse_args, get_checkpoint_path, parse_contexts
from DNN_def import build_network
from DNN_data_read import read_data_train, my_DataRead
import scipy.io as scio
from Error import get_err





if __name__ == '__main__':
	network = build_network()
	args = parse_args()
	args.config.write(sys.stdout)
	device = mx.gpu(0)
	switch = args.config.get('test', 'test')

	batch_size = args.config.getint('train', 'batch_size')
	mean_file = args.config.get('data', 'mean_file')
	xdim = args.config.getint('data', 'xdim')
	ydim = args.config.getint('data', 'ydim')
	test_input = args.config.get('test', 'test_input')
	test_target = args.config.get('test', 'test_target')
	

	test_prefix = args.config.get('test', 'test_prefix')

	load_epoch_num = args.config.getint('test', 'load_epoch_num')
	out_file = args.config.get('test', 'out_file')

	if switch.lower() == 'true':
		test_data_args = {
			"gpu_chunk": 32768,
			"input_file": test_input,
			"target_file": test_target,
			"mean_file": mean_file,
			"xdim": xdim,
			"ydim": ydim
			}
		test_sets = my_DataRead(test_data_args)
		(test_feats, test_tgts, test_ids, test_fr) = read_data_train(test_sets)

		max_feat = np.loadtxt("max_feat.txt")
		max_label = np.loadtxt("max_label.txt")
		min_label = np.loadtxt("min_label.txt")
		test_data = np.divide(test_feats, max_feat) - 0.5
		test_label = np.divide((test_tgts-min_label),(max_label-min_label))

		test=mx.io.NDArrayIter(test_data,  batch_size=batch_size, shuffle=False)

		pretrained_model = mx.model.FeedForward.load(test_prefix, load_epoch_num)

		model = mx.model.FeedForward(symbol=pretrained_model.symbol, arg_params=pretrained_model.arg_params, aux_params=pretrained_model.aux_params)
		outputs				= model.predict(
			X				  = test
			)
		
		

		result = np.multiply(outputs, (max_label-min_label))+min_label

		num = 0
		isExists=os.path.exists(out_file)
		if not isExists:
			 os.makedirs(out_file)
		for i in range(len(test_ids)):
			frame_num = test_fr[i]
			gen_data = result[num:(num+frame_num),:]
			num += frame_num
			
			np.savetxt(out_file+'/'+test_ids[i]+".txt", gen_data)

		(mcd,rmse,bap_err,vu_err) = get_err(test_tgts,result)
		f = open(out_file+"/err.txt", 'w')
		f.write( "mcd: %f \n" % mcd)
		f.write( "rmse: %f \n" % rmse)
		f.write( "bap_err: %f \n" % bap_err)
		f.write("vu: %f \n" % vu_err)
		f.close()






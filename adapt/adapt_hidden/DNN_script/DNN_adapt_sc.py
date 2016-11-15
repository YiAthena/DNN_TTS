import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import logging
import time
import os
from config_util import parse_args, get_checkpoint_path, parse_contexts
from DNN_def import sc_network
from DNN_data_read import read_data_train, my_DataRead
from Error import get_err

if __name__ == '__main__':

	
	args = parse_args()
	args.config.write(sys.stdout)
	device = mx.gpu(0)
	switch = args.config.get('adapt_sc', 'adapt_sc')

	if switch.lower() == 'true':
		load_prefix = args.config.get('adapt_sc', 'load_prefix')
		load_adapt_num = args.config.getint('adapt_sc', 'load_adapt_num')
		sc_prefix = args.config.get('adapt_sc', 'sc_prefix')

		num_epochs = args.config.getint('adapt_sc', 'num_epochs')

		batch_size = args.config.getint('train', 'batch_size')
		
		lr = args.config.getfloat('train', 'learning_rate')
		wd = args.config.getfloat('train', 'weight_decay')
		momentum = args.config.getfloat('train', 'momentum')
		decay_factor = args.config.getfloat('train', 'decay_factor')
		decay_lower_bound = args.config.getfloat('train', 'decay_lower_bound')

		train_input = args.config.get('adapt_sc', 'train_input')
		train_target = args.config.get('adapt_sc', 'train_target')
		val_input = args.config.get('adapt_sc', 'val_input')
		val_target = args.config.get('adapt_sc', 'val_target')
		xdim = args.config.getint('data', 'xdim')
		ydim = args.config.getint('data', 'ydim')

		train_num = args.config.getint('adapt_sc', 'train_num')
		val_num = args.config.getint('adapt_sc', 'val_num')

		data_name='data'
		label_name='softmax_label'

		train_data_args = {
			"gpu_chunk": 32768,
			"input_file": train_input,
			"target_file": train_target,
			"xdim": xdim,
			"ydim": ydim
			}
		train_sets=my_DataRead(train_data_args)
		(train_data, train_label, train_ids, train_frs) = read_data_train(train_sets, train_num)

		val_data_args={
			"gpu_chunk": 32768,
			"input_file": val_input,
			"target_file": val_target,
		  #  "mean_file": mean_file,
			"xdim": xdim,
			"ydim": ydim
			}
		val_sets=my_DataRead(val_data_args)
		(val_data, val_label, val_ids, val_frs) = read_data_train(val_sets, val_num)

		max_feat = np.loadtxt('max_feat.txt')
		max_label = np.loadtxt('max_label.txt')
		min_label = np.loadtxt('min_label.txt')

		train_data = np.divide(train_data, max_feat) - 0.5
		train_label = np.divide((train_label-min_label),(max_label-min_label))
		val_data= np.divide(val_data, max_feat) - 0.5
		val_label = np.divide((val_label-min_label),(max_label-min_label))

		train=mx.io.NDArrayIter(data = train_data, label=train_label, batch_size=batch_size, shuffle=False)
		val=mx.io.NDArrayIter(data= val_data,label=val_label,  batch_size=batch_size, shuffle=False)

		num_decay_iter = train_data.shape[0]/batch_size*10

	#	pretrained_model = mx.model.FeedForward.load(load_prefix, load_adapt_num)
		pretrained_model = mx.model.FeedForward.load(load_prefix, load_adapt_num)

		pre_arg_params=pretrained_model.arg_params

		fixed_names = ['fc1_weight','fc2_weight','fc3_weight','fc4_weight']
		fixed_arrays = [pre_arg_params[name] for name in fixed_names]
		fixed_param_names = dict(zip(fixed_names, fixed_arrays))

		fixed_param_names=pre_arg_params
	#	mod = mx.mod.Module(network, data_names='data', context=device, fixed_param_names=fixed_param_names)
		network = sc_network()
		mod = mx.mod.Module(network, context=device, fixed_param_names=fixed_param_names)
		mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label, for_training=True)
		
		
		mod.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=pre_arg_params, allow_missing=True)

		#lr_scheduler	   = mx.lr_scheduler.FactorScheduler(num_decay_iter, 1/decay_factor, stop_factor_lr=decay_lower_bound)

		mod.fit(train_data = train, eval_data=val, num_epoch=num_epochs, eval_metric = mx.metric.RMSE(),
				optimizer_params=(('learning_rate', 1), ('momentum', momentum)),
				epoch_end_callback = mx.callback.do_checkpoint(prefix=sc_prefix))

###############################################
	


###############################################
		test_input = args.config.get('adapt_sc', 'test_input')
		test_target = args.config.get('adapt_sc', 'test_target')
		out_file = args.config.get('adapt_sc', 'out_file')
		isExists=os.path.exists(out_file)
		if not isExists:
			 os.makedirs(out_file)
		test_data_args = {
			"gpu_chunk": 32768,
			"input_file": test_input,
			"target_file": test_target,
			"xdim": xdim,
			"ydim": ydim
			}
		test_sets = my_DataRead(test_data_args)
		(test_feats, test_tgts, test_ids, test_fr) = read_data_train(test_sets,20)
		max_feat = np.loadtxt("max_feat.txt")
		max_label = np.loadtxt("max_label.txt")
		min_label = np.loadtxt("min_label.txt")
		test_data = np.divide(test_feats, max_feat) - 0.5
		test_label = np.divide((test_tgts-min_label),(max_label-min_label))

		test=mx.io.NDArrayIter(test_data,  batch_size=batch_size, shuffle=False)

		sc_model = mx.model.FeedForward.load(sc_prefix, num_epochs)
		model = mx.model.FeedForward(symbol=sc_model.symbol, arg_params=sc_model.arg_params, aux_params=sc_model.aux_params)
		outputs				= model.predict(
			X				  = test
			)

		result = np.multiply(outputs, (max_label-min_label))+min_label
		print result
		num = 0
		print test_ids
		for i in range(len(test_ids)):
			frame_num = test_fr[i]
			print test_fr[i]
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



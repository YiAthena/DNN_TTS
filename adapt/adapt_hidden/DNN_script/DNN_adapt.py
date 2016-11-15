import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import logging
import time

from config_util import parse_args, get_checkpoint_path, parse_contexts
from DNN_def import adapt_network
from DNN_data_read import read_data_train, my_DataRead

if __name__ == '__main__':
	network = adapt_network()

	args = parse_args()
	args.config.write(sys.stdout)
	device = mx.gpu(0)
	switch = args.config.get('adapt', 'adapt')

	if switch.lower() == 'true':

		adapt_prefix = args.config.get('adapt', 'adapt_prefix')
		load_prefix = args.config.get('adapt', 'load_prefix')
		load_adapt_num = args.config.getint('adapt', 'load_adapt_num')

		batch_size = args.config.getint('train', 'batch_size')
		num_epochs = args.config.getint('train', 'num_epoch')
		lr = args.config.getfloat('train', 'learning_rate')
		wd = args.config.getfloat('train', 'weight_decay')
		momentum = args.config.getfloat('train', 'momentum')
		decay_factor = args.config.getfloat('train', 'decay_factor')
		decay_lower_bound = args.config.getfloat('train', 'decay_lower_bound')

		

		train_input = args.config.get('data', 'train_input')
		train_target = args.config.get('data', 'train_target')
		val_input = args.config.get('data', 'val_input')
		val_target = args.config.get('data', 'val_target')
		
		xdim = args.config.getint('data', 'xdim')
		ydim = args.config.getint('data', 'ydim')

		data_name='data'
		label_name='softmax_label'

		train_input_all = train_input.split(',')
		train_target_all = train_target.split(',')
		len_train = len(train_input_all)
		assert len(train_input_all) == len(train_target_all), 'input_path_len != target_path_len'

		val_input_all = val_input.split(',')
		val_target_all = val_target.split(',')
		len_dev = len(val_target_all)
		assert len(val_input_all) == len(val_target_all), 'input_path_len != target_path_len'

		train_data_args = []
		val_data_args = []

		train_data = []
		train_label = []
		train_data = np.empty(shape=[0, xdim])
		train_label = np.empty(shape=[0,ydim])
		train_ids =[]
		train_frs = []
		train_sc = []
		train_sc = np.empty(shape=[0, 4])	
		for i in range(len_train):
			train_data_args={
			"gpu_chunk" : 32768,
			"input_file" : train_input_all[i],
			"target_file" : train_target_all[i],
		   # "mean_file" : mean_file,
			"xdim": xdim,
			"ydim": ydim
			}
			train_sets=my_DataRead(train_data_args)
			(feats, tgts, utt_ids, num_fr) = read_data_train(train_sets)
			train_data = np.concatenate((train_data, feats), axis=0)
			train_label = np.concatenate((train_label, tgts), axis=0)
			train_ids.append(utt_ids)
			train_frs.append(num_fr)
			sc_tmp = np.zeros((sum(num_fr),4))
			sc_tmp[:,i] = 1
			train_sc = np.concatenate((train_sc,sc_tmp), axis=0)

		val_data = []
		val_label = []
		val_data = np.empty(shape=[0, xdim])
		val_label = np.empty(shape=[0,ydim])
		val_ids =[]
		val_frs = []
		val_sc = []
		val_sc = np.empty(shape=[0, 4])	
		for i in range(len_dev):
			val_data_args={
			"gpu_chunk": 32768,
			"input_file": val_input_all[i],
			"target_file": val_target_all[i],
		  #  "mean_file": mean_file,
			"xdim": xdim,
			"ydim": ydim
			}

			val_sets=my_DataRead(val_data_args)
			(feats, tgts, utt_ids, num_fr) = read_data_train(val_sets)
			val_data = np.append(val_data, feats, axis=0)
			val_label = np.append(val_label, tgts, axis=0)
			val_ids.append(utt_ids)
			val_frs.append(num_fr)
			sc_tmp = np.zeros((sum(num_fr),4))
			sc_tmp[:,i] = 1
			val_sc = np.concatenate((val_sc,sc_tmp), axis=0)


		max_feat = np.loadtxt('max_feat.txt')
		max_label = np.loadtxt('max_label.txt')
		min_label = np.loadtxt('min_label.txt')

		train_data = np.divide(train_data, max_feat) - 0.5
		train_label = np.divide((train_label-min_label),(max_label-min_label))
		val_data= np.divide(val_data, max_feat) - 0.5
		val_label = np.divide((val_label-min_label),(max_label-min_label))


		train=mx.io.NDArrayIter({'data': train_data, 'sc': train_sc}, label=train_label, batch_size=batch_size, shuffle=False)
		val=mx.io.NDArrayIter({'data': val_data, 'sc': val_sc},label=val_label,  batch_size=batch_size, shuffle=False)

		num_decay_iter = train_data.shape[0]/batch_size*10

		#load_model
		pretrained_model = mx.model.FeedForward.load(load_prefix, load_adapt_num)
		pre_arg_params=pretrained_model.arg_params
		#print pre_arg_params.keys()
		#print pre_arg_params.values()
		#delete not fixed paras  
		#pre_arg_params['sc'] = mx.ndarray.zeros(4)
		#pre_arg_params['sc'] = mx.nd.zeros((batch_size, 4))
		#mod = mx.mod.Module(network, data_names=('data', 'sc'), context=device, work_load_list=None, fixed_param_names=pre_arg_params.keys())
		mod = mx.mod.Module(network, data_names=('data', 'sc'), context=device, fixed_param_names=pre_arg_params)
		mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label, for_training=True)
		
		#mod.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=pre_arg_params, allow_missing=True)

		mod.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=pre_arg_params, allow_missing=True)

		lr_scheduler	   = mx.lr_scheduler.FactorScheduler(num_decay_iter, 1/decay_factor, stop_factor_lr=decay_lower_bound)
		#('lr_scheduler', lr_scheduler), , ('wd', wd))


		mod.fit(train_data = train, eval_data=val, num_epoch=num_epochs, eval_metric = mx.metric.RMSE(),
				optimizer_params=(('learning_rate', lr), ('momentum', momentum),('lr_scheduler', lr_scheduler),('wd', wd)),
				epoch_end_callback = mx.callback.do_checkpoint(prefix=adapt_prefix))

		# model = mx.model.FeedForward(
		#  	ctx				= device,
		# 	symbol			 = network,
		# 	num_epoch		  = num_epochs,
		# 	learning_rate	  = lr,
		# 	lr_scheduler	   = mx.lr_scheduler.FactorScheduler(num_decay_iter, 1/decay_factor, stop_factor_lr=decay_lower_bound),
		# 	momentum		   = momentum,
		# 	wd				 = wd,
		# 	initializer		= mx.init.Xavier(factor_type="in", magnitude=2.34))

		# mod.fit(
		# 	X				  = train,
		# 	eval_data		  = val,
		# 	eval_metric		= mx.metric.RMSE(),
		# 	epoch_end_callback = mx.callback.do_checkpoint(prefix=adapt_prefix))







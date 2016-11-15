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

if __name__ == '__main__':
	network = build_network()
	args = parse_args()
	args.config.write(sys.stdout)
	device = mx.gpu(0)
	switch = args.config.get('train', 'train')

	if switch.lower() == 'true':
		batch_size = args.config.getint('train', 'batch_size')
		num_epochs = args.config.getint('train', 'num_epoch')
		lr = args.config.getfloat('train', 'learning_rate')
		wd = args.config.getfloat('train', 'weight_decay')
		momentum = args.config.getfloat('train', 'momentum')
		decay_factor = args.config.getfloat('train', 'decay_factor')
		decay_lower_bound = args.config.getfloat('train', 'decay_lower_bound')
		#num_decay_iter = args.config.getint('train', 'num_decay_iter')

		model_prefix = args.config.get('data', 'model_prefix')
		train_input = args.config.get('data', 'train_input')
		train_target = args.config.get('data', 'train_target')
		val_input = args.config.get('data', 'val_input')
		val_target = args.config.get('data', 'val_target')
		mean_file = args.config.get('data', 'mean_file')
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

		val_data = []
		val_label = []
		val_data = np.empty(shape=[0, xdim])
		val_label = np.empty(shape=[0,ydim])
		val_ids =[]
		val_frs = []
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

		# train_sets = my_DataRead(train_data_args)
		# val_sets = my_DataRead(val_data_args)
		# (train_data, train_label, train_ids) = read_data_train(train_sets, xdim, ydim)
		# (val_data, val_label, val_ids) = read_data_train(val_sets, xdim, ydim)
		#(train_data, train_label, train_ids, train_framenum, sc_train) = read_data(train_input, train_target, xdim, ydim)
		#(val_data, val_label, val_ids, val_framenum, sc_val) = read_data(val_input, val_target, xdim, ydim)


		
###########################
#max norm
		# max_feat = np.amax(train_data, axis=0)
		# max_feat[max_feat<1] = 1

		# max_label = np.amax(train_label, axis=0)
		# min_label = np.amin(train_label, axis=0)

		# np.savetxt('max_feat.txt',max_feat, delimiter=' ')
		# np.savetxt('max_label.txt',max_label, delimiter=' ')
		# np.savetxt('min_label.txt',min_label, delimiter=' ')

		# #print (train_data.shape)
		# train_data = np.divide(train_data, max_feat) - 0.5
		# train_label = np.divide((train_label-min_label),(max_label-min_label))
		# val_data= np.divide(val_data, max_feat) - 0.5
		# val_label = np.divide((val_label-min_label),(max_label-min_label))
###########################################
#mean norm

		input_mean= scio.loadmat(mean_file+"/input_mean.mat").values()[0]
   		input_std= scio.loadmat(mean_file+"/input_std.mat").values()[0]
		output_mean= scio.loadmat(mean_file+"/output_mean.mat").values()[0]
		output_std= scio.loadmat(mean_file+"/output_std.mat").values()[0]

		train_data = np.divide((train_data - input_mean),input_std)
		train_label = np.divide((train_label - output_mean),output_std)
		val_data = np.divide((val_data - input_mean),input_std)
		val_label = np.divide((val_label - output_mean),output_std)

#############################################
		#(train_data, train_label) = norm_data(train_data, train_label, mean_file)
		#(val_data, train_label) = norm_data(val_data, train_label, mean_file)

		train=mx.io.NDArrayIter(train_data, label=train_label, batch_size=batch_size, shuffle=True)
		val=mx.io.NDArrayIter(val_data,label=val_label,  batch_size=batch_size, shuffle=True)

		# print train_data.shape
		# print train_label.shape
		num_decay_iter = train_data.shape[0]/batch_size*30
		model = mx.model.FeedForward(
		 	ctx				= device,
			symbol			 = network,
			num_epoch		  = num_epochs,
			learning_rate	  = lr,
		#	lr_scheduler	   = mx.lr_scheduler.FactorScheduler(num_decay_iter, 1/decay_factor, stop_factor_lr=decay_lower_bound),
			momentum		   = momentum,
		#	wd				 = wd,
		#	initializer		= mx.init.Xavier(factor_type="in", magnitude=0.05))
			initializer		= mx.init.Xavier(rnd_type="uniform", factor_type="in", magnitude=2.34))
		model.fit(
			X				  = train,
			eval_data		  = val,
			eval_metric		= mx.metric.RMSE(),
			epoch_end_callback = mx.callback.do_checkpoint(prefix=model_prefix))
#########################################################################
		test_input = args.config.get('test', 'test_input')
		test_target = args.config.get('test', 'test_target')
		out_file = args.config.get('test', 'out_file')
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

		test_data = np.divide(test_feats, max_feat) - 0.5
		test_label = np.divide((test_tgts-min_label),(max_label-min_label))

		test=mx.io.NDArrayIter(test_data,  batch_size=batch_size, shuffle=False)

		outputs				= model.predict(
			X				  = test
			)

		#result = np.multiply(outputs, (max_label-min_label))+min_label
		result = np.multiply(outputs, output_std)+output_mean

		num = 0

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




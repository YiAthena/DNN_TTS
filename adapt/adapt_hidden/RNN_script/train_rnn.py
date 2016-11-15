import re
import sys
sys.path.insert(0, "../../python")
import time
import logging
import os.path

import mxnet as mx
import numpy as np
from speechSGD import speechSGD
from hybrid import hybrid_net, lstm_unroll

from io_util import TruncatedSentenceIter
from config_util import parse_args, get_checkpoint_path, parse_contexts
from load_data import my_DataRead

METHOD_TBPTT = 'truncated-bptt'

def prepare_data(args):
	batch_size = args.config.getint('train', 'batch_size')
	num_hidden_lstm = args.config.getint('arch', 'num_hidden_lstm')
	num_lstm_layer = args.config.getint('arch', 'num_lstm_layer')

	init_c = [('l%d_init_c'%l, (batch_size, num_hidden_lstm)) for l in range(num_lstm_layer)]
	
	init_h = [('l%d_init_h'%l, (batch_size, num_hidden_lstm)) for l in range(num_lstm_layer)]

	init_states = init_c + init_h

	train_input = args.config.get('data', 'train_input')
	train_target = args.config.get('data', 'train_target')
	val_input = args.config.get('data', 'val_input')
	val_target = args.config.get('data', 'val_target')
	xdim = args.config.getint('data', 'xdim')
	ydim = args.config.getint('data', 'ydim')

	train_input_all = train_input.split(',')
	train_target_all = train_target.split(',')
	len_train = len(train_input_all)
	assert len(train_input_all) == len(train_target_all), 'input_path_len != target_path_len'

	val_input_all = val_input.split(',')
	val_target_all = val_target.split(',')
	len_val = len(val_target_all)
	assert len(val_input_all) == len(val_target_all), 'input_path_len != target_path_len'

	train_sets=[]
	val_sets = []

	for i in range(len_train):
		train_data_args={
		"gpu_chunk" : 32768,
		"input_file" : train_input_all[i],
		"target_file" : train_target_all[i],
	  	"xdim": xdim,
		"ydim": ydim
		}
		train_sets.append(my_DataRead(train_data_args))

	for i in range(len_val):
		val_data_args={
		"gpu_chunk": 32768,
		"input_file": val_input_all[i],
		"target_file": val_target_all[i],
		"xdim": xdim,
		"ydim": ydim
		}
		val_sets.append(my_DataRead(val_data_args))
	return (init_states, train_sets, val_sets)

def RMSE(labels, preds):
	labels = labels.reshape((-1,labels.shape[2]))
	preds = preds.reshape((-1, preds.shape[2]))
	loss = 0.
	num_inst = 0
	for i in range(preds.shape[0]):
		loss += np.sqrt(((labels[i]-preds[i])**2).mean())
		num_inst += 1
	loss=loss/preds.shape[0]
	return loss , num_inst

def SSE(labels, preds):

	labels = labels.reshape((-1,labels.shape[2]))
	preds = preds.reshape((-1, preds.shape[2]))
	loss = 0.
	num_inst = 0
	
	for i in range(preds.shape[0]):
		loss += sum((labels[i]-preds[i])**2)
		num_inst += 1
	loss=loss/num_inst
	return loss , num_inst

def Acc_exclude_padding(labels, preds):
	labels = labels.reshape((-1,))
	preds = preds.reshape((-1, preds.shape[2]))
	sum_metric = 0
	num_inst = 0
	for i in range(preds.shape[0]):
		pred_label = np.argmax(preds[i], axis=0)
		label = labels[i]

		ind = np.nonzero(label.flat)
		pred_label_real = pred_label.flat[ind]
		label_real = label.flat[ind]
		sum_metric += (pred_label_real == label_real).sum()
		num_inst += len(pred_label_real)
	return sum_metric, num_inst 

class SimpleLRScheduler(mx.lr_scheduler.LRScheduler):
	def __init__(self, dynamic_lr, effective_sample_count=1, momentum=0.9, optimizer="sgd"):
		super(SimpleLRScheduler, self).__init__()
		self.dynamic_lr = dynamic_lr
		self.effective_sample_count = effective_sample_count
		self.momentum = momentum
		self.optimizer = optimizer

	def __call__(self, num_update):
		if self.optimizer == "speechSGD":
			return self.dynamic_lr / self.effective_sample_count, self.momentum
		else:
			return self.dynamic_lr / self.effective_sample_count

def score_with_state_forwarding(module, eval_data, eval_metric):
	eval_data.reset()
	eval_metric.reset()

	for eval_batch in eval_data:
		module.forward(eval_batch, is_train=False)
		module.update_metric(eval_metric, eval_batch.label)

		# copy over states
		outputs = module.get_outputs()
		# outputs[0] is softmax, 1:end are states
		for i in range(1, len(outputs)):
			outputs[i].copyto(eval_data.init_state_arrays[i-1])


def get_initializer(args):
	init_type = getattr(mx.initializer, args.config.get('train', 'initializer'))
	init_scale = args.config.getfloat('train', 'init_scale')
	if init_type is mx.initializer.Xavier:
		return mx.initializer.Xavier(magnitude=init_scale)
	return init_type(init_scale)

def do_training(training_method, args, module, data_train, data_val):
	from distutils.dir_util import mkpath
	mkpath(os.path.dirname(get_checkpoint_path(args)))

	batch_size = data_train.batch_size
	batch_end_callbacks = [mx.callback.Speedometer(batch_size, 
												   args.config.getint('train', 'show_every'))]
	eval_allow_extra = True if training_method == METHOD_TBPTT else False
		#	eval_metric = [mx.metric.np(CrossEntropy, allow_extra_outputs=eval_allow_extra),
	#eval_metric = [mx.metric.np(RMSE, allow_extra_outputs=eval_allow_extra),
	#			   mx.metric.np(Acc_exclude_padding, allow_extra_outputs=eval_allow_extra)]
	eval_metric = [mx.metric.np(RMSE, allow_extra_outputs=eval_allow_extra)]
	eval_metric = mx.metric.create(eval_metric)
		#eval_metric = mx.metric.RMSE()
	optimizer = args.config.get('train', 'optimizer')
	momentum = args.config.getfloat('train', 'momentum')
	learning_rate = args.config.getfloat('train', 'learning_rate')
	lr_scheduler = SimpleLRScheduler(learning_rate, momentum=momentum, optimizer=optimizer)

	if training_method == METHOD_TBPTT:
		lr_scheduler.seq_len = data_train.truncate_len

	n_epoch = 0
	num_epoch = args.config.getint('train', 'num_epoch')
	learning_rate = args.config.getfloat('train', 'learning_rate')
	decay_factor = args.config.getfloat('train', 'decay_factor')
	decay_bound = args.config.getfloat('train', 'decay_lower_bound')
	clip_gradient = args.config.getfloat('train', 'clip_gradient')
	weight_decay = args.config.getfloat('train', 'weight_decay')
	if clip_gradient == 0:
		clip_gradient = None

	last_acc = -float("Inf")
	last_params = None

	module.bind(data_shapes=data_train.provide_data,
				label_shapes=data_train.provide_label,
				for_training=True)
	module.init_params(initializer=get_initializer(args))

	def reset_optimizer():
		if optimizer == "sgd" or optimizer == "speechSGD":
			module.init_optimizer(kvstore='local',
							  optimizer=args.config.get('train', 'optimizer'),
							  optimizer_params={'lr_scheduler': lr_scheduler,
												'momentum': momentum,
												'rescale_grad': 1.0,
												'clip_gradient': clip_gradient,
												'wd': weight_decay},
							  force_init=True)
		else:
			module.init_optimizer(kvstore='local',
							  optimizer=args.config.get('train', 'optimizer'),
							  optimizer_params={'lr_scheduler': lr_scheduler,
												'rescale_grad': 1.0,
												'clip_gradient': clip_gradient,
												'wd': weight_decay},
							  force_init=True)
	reset_optimizer()

	while True:
		tic = time.time()
		eval_metric.reset()

		for nbatch, data_batch in enumerate(data_train):
			if training_method == METHOD_TBPTT:
				lr_scheduler.effective_sample_count = data_train.batch_size * truncate_len
				lr_scheduler.momentum = np.power(np.power(momentum, 1.0/(data_train.batch_size * truncate_len)), data_batch.effective_sample_count)
			else:
				if data_batch.effective_sample_count is not None:
					lr_scheduler.effective_sample_count = data_batch.effective_sample_count

			module.forward_backward(data_batch)
			module.update()
			module.update_metric(eval_metric, data_batch.label)

			batch_end_params = mx.model.BatchEndParam(epoch=n_epoch, nbatch=nbatch,
													  eval_metric=eval_metric,
													  locals=None)
			for callback in batch_end_callbacks:
				callback(batch_end_params)

			if training_method == METHOD_TBPTT:
				# copy over states
				outputs = module.get_outputs()
				# outputs[0] is softmax, 1:end are states
				for i in range(1, len(outputs)):
					outputs[i].copyto(data_train.init_state_arrays[i-1])

		# for name, val in eval_metric.get_name_value():
		# 	logging.info('Epoch[%d] Train-%s=%f', n_epoch, name, val)
		
		val = eval_metric.get_name_value()[0][1]
		logging.info('Epoch[%d] Train-%s=%f', n_epoch, 'RMSE', val)
	#	toc = time.time()
	#	logging.info('Epoch[%d] Time cost=%.3f', n_epoch, toc-tic)

		data_train.reset()

		# test on eval data
		score_with_state_forwarding(module, data_val, eval_metric)

		# test whether we should decay learning rate
		curr_acc = None
		# for name, val in eval_metric.get_name_value():
		# 	logging.info("Epoch[%d] Dev-%s=%f", n_epoch, name, val)
		# 	if name == 'SSE':
		# 		curr_acc = val
		val = eval_metric.get_name_value()[0][1]
		curr_acc = val
		logging.info("Epoch[%d] Dev-%s=%f", n_epoch, 'RMSE', val)
		assert curr_acc is not None, 'cannot find Acc_exclude_padding in eval metric'

		if n_epoch > 0 and lr_scheduler.dynamic_lr > decay_bound and curr_acc > last_acc:
			logging.info('Epoch[%d] !!! Dev set performance drops, reverting this epoch',
						 n_epoch)
			logging.info('Epoch[%d] !!! LR decay: %g => %g', n_epoch,
						 lr_scheduler.dynamic_lr, lr_scheduler.dynamic_lr / float(decay_factor))

			lr_scheduler.dynamic_lr /= decay_factor
			# we reset the optimizer because the internal states (e.g. momentum)
			# might already be exploded, so we want to start from fresh
			reset_optimizer()
			module.set_params(*last_params)
		else:
			last_params = module.get_params()
			last_acc = curr_acc
			n_epoch += 1

			# save checkpoints
			mx.model.save_checkpoint(get_checkpoint_path(args), n_epoch,
									 module.symbol, *last_params)

		if n_epoch == num_epoch:
			break


if __name__ == '__main__':
	args = parse_args()
	args.config.write(sys.stdout)

	training_method = args.config.get('train', 'method')
	batch_size = args.config.getint('train', 'batch_size')
	dropout = args.config.getfloat('train', 'dropout')

	feat_dim = args.config.getint('data', 'xdim')
	label_dim = args.config.getint('data', 'ydim')

	num_ff_layer = args.config.getint('arch', 'num_ff_layer')
	num_hidden_ff = args.config.getint('arch', 'num_hidden_ff')
	num_lstm_layer = args.config.getint('arch', 'num_lstm_layer')
	num_hidden_lstm = args.config.getint('arch', 'num_hidden_lstm')
	num_epoch = args.config.getint('train', 'num_epoch')

	momentum = args.config.getfloat('train', 'momentum')
	lr = args.config.getfloat('train', 'learning_rate')

	weight_decay = args.config.getfloat('train', 'weight_decay')

	contexts = parse_contexts(args)

	init_states, train_sets, val_sets = prepare_data(args)
   
	state_names = [x[0] for x in init_states]

	logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

	
	if training_method == METHOD_TBPTT:
		truncate_len = args.config.getint('train', 'truncate_len')
		data_train = TruncatedSentenceIter(train_sets, batch_size, init_states,truncate_len=truncate_len, delay=10, feat_dim=feat_dim, label_dim=label_dim, data_name='data', label_name='linear_label', do_shuffling=True, pad_zeros=False)
		data_val = TruncatedSentenceIter(val_sets,batch_size, init_states,truncate_len=truncate_len, delay=10, feat_dim=feat_dim, label_dim=label_dim, data_name='data', label_name='linear_label', do_shuffling=True, pad_zeros=False)
		
		sym = hybrid_net(num_ff_layer, num_hidden_ff, num_lstm_layer, num_hidden_lstm, seq_len=truncate_len, num_label=label_dim)
		#sym = lstm_unroll(num_lstm_layer, seq_len=truncate_len, num_hidden_lstm=num_hidden_lstm, num_label=label_dim, dropout=0.2)
		data_names = [x[0] for x in data_train.provide_data]
		label_names = [x[0] for x in data_train.provide_label]
		module = mx.mod.Module(sym, context=contexts, data_names=data_names,label_names=label_names)
		do_training(training_method, args, module, data_train, data_val)
		# mod = mx.mod.Module(sym, context=contexts, data_names=data_names,label_names=label_names)
		# mod.bind(data_shapes=data_train.provide_data, label_shapes=data_val.provide_label, for_training=True)


		# model_prefix = args.config.get('train', 'model_prefix')
		# isExists=os.path.exists(os.path.dirname(model_prefix))
		# if not isExists:
		# 	 os.makedirs(os.path.dirname(model_prefix))
		

		# #('wd', wd)
		# mod.fit(train_data = data_train, eval_data=data_val, num_epoch=num_epoch, eval_metric = mx.metric.RMSE(),
		# 		optimizer_params=(('learning_rate', lr), ('momentum', momentum)),
		# 		epoch_end_callback = mx.callback.do_checkpoint(prefix=model_prefix))

	else:
		raise RuntimeError('Unknown training method: %s' % training_method)



	print("="*80)
	print("Finished Training")
	print("="*80)
	args.config.write(sys.stdout)



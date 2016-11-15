import os
def gen_cfg_train(cfg_file, stru, nodes, net_path, nc_path, learning_rate, momentum, parallel_sequences):
	autosave = net_path + '/autosave'
	
	tmp = net_path + '/tmp'

	if not os.path.exists(autosave):
		os.makedirs(autosave)
	
	if not os.path.exists(tmp):
		os.makedirs(tmp)

	train_nc = nc_path + '/train.nc'
	val_nc = nc_path + '/val.nc'
	test_nc = nc_path + '/test.nc'
	jsn_path = net_path + '/jsn'
	jsn_file = jsn_path + '/'+ stru + '_' +nodes + '.jsn'


	buf = open(cfg_file, 'w')  
	print >>buf, "network = %s" %(jsn_file)
	print >>buf, "train_file = %s" %(train_nc)
	print >>buf, "test_file = %s" %(test_nc)
	print >>buf, "val_file = %s" %(val_nc)
	print >>buf, "save_network = %s" %(autosave + '/trained_' + stru + nodes)
	print >>buf, "autosave_prefix = %s" %(autosave + '/' +stru + nodes)
	print >>buf, "cache_path = %s" %(tmp)

	print >>buf, "learning_rate = %f" %(learning_rate)
	print >>buf, "momentum = %f" %(momentum)

	print >>buf, "cuda = true"
	print >>buf, "list_devices = false"
	print >>buf, "parallel_sequences = %d" %(parallel_sequences)
	print >>buf, "random_seed = 0"
	print >>buf, "train = true"
	print >>buf, "stochastic = true"
	print >>buf, "shuffle_fractions = true"
	print >>buf, "shuffle_sequences = false"
	print >>buf, "max_epochs = -1"
	print >>buf, "max_epochs_no_best = 50"
	print >>buf, "validate_every = 1"
	print >>buf, "test_every = 5"
	print >>buf, "optimizer = steepest_descent"
	
	print >>buf, "autosave = true"
	print >>buf, "autosave_best = false"
	print >>buf, "train_fraction = 1"
	print >>buf, "test_fraction = 1"
	print >>buf, "val_fraction = 1"
	print >>buf, "truncate_seq = 0"
	print >>buf, "input_noise_sigma = 0.1"
	print >>buf, "input_right_context = 0"
	print >>buf, "input_left_context = 0"
	print >>buf, "output_time_lag = 0"

	print >>buf, "weight_noise_sigma = 0.0"
	print >>buf, "weights_dist = normal"
	print >>buf, "weights_normal_mean = 0"
	print >>buf, "weights_normal_sigma = 0.1"

	buf.close()

def get_type(mark):
	if mark == 'F':
		name = 'feedforward_tanh'
	elif mark == 'B':
		name = 'blstm'
	elif mark == 'U':
		name = 'lstm'
	else:
		print "unknow char"
		exit ()
	return name


def gen_jsn(jsn_file, stru, nodes, dim_input, dim_target):
	nodes = nodes.split('_')
	assert len(nodes) == len(stru), "net structure error"
	num_layers = len(stru)

	buf = open(jsn_file, 'w') 
	print >>buf, "{"
	print >>buf, "\t\"layers\":["

	print >>buf, "\t{"
	print >>buf, "\t\t\"size\":%d," %(dim_input)
	print >>buf, "\t\t\"name\":\"input\","
	print >>buf, "\t\t\"type\":\"input\""
	print >>buf, "\t},"

	for i in range(num_layers):
		print >>buf, "\t{"
		print >>buf, "\t\t\"size\":%d," %(int(nodes[i]))
		print >>buf, "\t\t\"name\":\"%s_%d\"," %(get_type(stru[i]),i)
		print >>buf, "\t\t\"bias\":1.0,"
		print >>buf, "\t\t\"type\":\"%s\"" % (get_type(stru[i]))
		print >>buf, "\t},"

	print >>buf, "\t{"
	print >>buf, "\t\t\"size\":%d," %(dim_target)
	print >>buf, "\t\t\"name\":\"output\","
	print >>buf, "\t\t\"bias\":1.0,"
	print >>buf, "\t\t\"type\":\"feedforward_identity\""
	print >>buf, "\t},"

	print >>buf, "\t{"
	print >>buf, "\t\t\"size\":%d," %(dim_target)
	print >>buf, "\t\t\"name\":\"postoutput\","
	print >>buf, "\t\t\"type\":\"sse\""
	print >>buf, "\t}"

	print >>buf, "\t]"
	print >>buf, "}"
	buf.close()

def gen_sh_train(train_file, cfg_file):
	buf = open(train_file, 'w')
	print >>buf, "#!/bin/bash"
	print >>buf, "currennt\t--options_file %s" %(cfg_file)
	buf.close()







	
	
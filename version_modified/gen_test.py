from config_util import parse_args
import os
import sys

def gen_test_cfg(test_nc, trained_jsn, test_cfg, csv_path, tmp):
	
	buf = open(test_cfg, 'w')  
	print >>buf, "train = false"
	print >>buf, "ff_output_format = csv"
	print >>buf, "ff_input_file = %s" %(test_nc)
	print >>buf, "parallel_sequences = 10"
	print >>buf, "cache_path = %s" %(tmp)
	print >>buf, "revert_std = false"
	print >>buf, "input_left_context = 0"
	print >>buf, "input_right_context = 0"
	print >>buf, "network = %s" %(trained_jsn)
	print >>buf, "ff_output_file= %s" %(csv_path)
	buf.close()

def gen_sh_train(test_file, test_cfg):
	buf = open(test_file, 'w')
	print >>buf, "#!/bin/bash"
	print >>buf, "currennt\t--options_file %s" %(test_cfg)
	buf.close()
	os.system("chmod +x %s" %(test_file))
	os.system("bash %s" %(test_file))

def csv2txt(csv_path, mean_path, gen_par_dyn, dim_target):
	buf = open('run_csv2txt.sh', 'w')  
	print >>buf, "csv_path=%s" %(csv_path)
	print >>buf, "mean_path=%s" %(mean_path)
	print >>buf, "gen_par_dyn=%s" %(gen_par_dyn)
	print >>buf, "dim_target=%d" %(dim_target) 
	print >>buf, "matlab -nojvm -nosplash -nodesktop -r \"file_input='$csv_path', mean_file='$mean_path',file_output='$gen_par_dyn',dim_target=$dim_target;csv2txt;quit\""
	buf.close()
	os.system("bash run_csv2txt.sh")

def postprocess(gv_path, gen_par_dyn, gen_par, mean_path, dim_target):
	buf = open('run_postprocess.sh', 'w')  
	print >>buf, "gv_path=%s" %(gv_path)
	print >>buf, "mean_path=%s" %(mean_path)
	print >>buf, "gen_par_dyn=%s" %(gen_par_dyn)
	print >>buf, "gen_par=%s" %(gen_par)
	print >>buf, "dim_target=%d" %(dim_target) 
	print >>buf, "matlab -nojvm -nosplash -nodesktop -r \"gv_file='$gv_path', mean_file='$mean_path',gen_par_dyn='$gen_par_dyn',gen_par='$gen_par',dim_target=$dim_target;postprocess;quit\""
	buf.close()
	os.system("bash run_postprocess.sh")


def wav_syn(gen_par, wav_path, dim_target):
	buf = open('run_wav_syn.sh', 'w')
	print >>buf, "gen_par=%s" %(gen_par)
	print >>buf, "wav_path=%s" %(wav_path)
	print >>buf, "dim_target=%d" %(dim_target) 
	print >>buf, "matlab -nojvm -nosplash -nodesktop -r \"gen_par='$gen_par',wav_path='$wav_path',dim_target=$dim_target;wav_syn;quit\""
	buf.close()
	os.system("bash run_wav_syn.sh")
	

if __name__ == '__main__':
	args = parse_args()
	args.config.write(sys.stdout)

	dim_target = args.config.getint('data', 'dim_target')
	nc_path = args.config.get('net', 'nc_path')
	test_nc = nc_path + '/test.nc' ###
	trained_jsn = args.config.get('test', 'trained_jsn')
	
	net_path = args.config.get('net', 'net_path')
	cfg_path = net_path +  '/cfg'
	test_cfg = cfg_path +'/' + 'test_' + os.path.basename(trained_jsn).split('.')[0] + '.cfg'
	tmp = net_path + '/tmp'

	result_path = args.config.get('test', 'result_path')
	result_path = result_path + '/' + os.path.basename(trained_jsn).split('.')[0]
	csv_path = result_path + '/csv'
	if not os.path.exists(csv_path):
		os.makedirs(csv_path)

	#gen_test_cfg(test_nc, trained_jsn,test_cfg, csv_path, tmp)

	sh_path = net_path + '/sh'
	test_file = sh_path + '/test_' + os.path.basename(trained_jsn).split('.')[0] + '.sh'
	#gen_sh_train(test_file, test_cfg)
	

	mean_path = args.config.get('feature', 'mean_path')
	gen_par_dyn = result_path + '/gen_par_dyn'
	if not os.path.exists(gen_par_dyn):
		os.makedirs(gen_par_dyn)
	#csv2txt(csv_path, mean_path, gen_par_dyn, dim_target)
###########calculate error#######################################
	
###########postprocess#######################################
	gv_path = args.config.get('feature', 'gv_path')
	gen_par = result_path + '/gen_par'
	if not os.path.exists(gen_par):
		os.makedirs(gen_par)
	postprocess(gv_path, gen_par_dyn, gen_par, mean_path, dim_target)

###########gen_wav#######################################
	wav_path = result_path + '/wav'
	if not os.path.exists(wav_path):
		os.makedirs(wav_path)
	wav_syn(gen_par, wav_path, dim_target)












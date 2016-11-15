import sys
import re
import os
import shutil
from config_util import parse_args
import numpy as np
import random
from gen_net_files import *

delimiter = ["^", "-", "+", "=", "@", "_", 
"/A:", "_", "_",
"/B:", "-", "-", "@", "-", "&", "-", "#", "-", "$", "-", "!", "-", ";", "-", "|",
"/C:", "+", "+",
"/D:", "_",
"/E:", "+", "@", "+", "&", "+", "#", "+",
"/F:", "_",
"/G:", "_",
"/H:", "@", "@", "=", "|",
"/I:", "=",
"/J:", "+", "-",
"[", "]" ]

phoneme_list = ['0','aau','ah','au','b','c','ch','d','eau','eengu',
'eenu','eeu','eh','eru','eu','f','g','h','iu','izhu','izu',
'j','k','l','m','n','ngu','nu','oh','oou','ou','p','q','r','s',
'sh','sil','sp','t','uu','vh','vu','wh','x','yh','z','zh' ]

tune_list = ['x', 'xx', '1', '2', '3', '4', '5']
boundary_list = ['x', '0', '1', '3', '4']
zero_list = ['x', '1', '2']

def lab_cats(nums):
	cats_array = [0 for i in range(nums)]
	pho_pos = [1,2,3]
	del_pos = range(24, 29) + range(46, 53)
	del_pos = [0,4,7,11] + del_pos
	tun_pos = [8,9,10]
	bou_pos = [29, 30]
	zer_pos = [6, 23]
	for i in pho_pos:
		cats_array[i] = 1
	for i in del_pos:
		cats_array[i] = 2
	for i in tun_pos:
		cats_array[i] = 3
	for i in bou_pos:
		cats_array[i] = 4
	for i in zer_pos:
		cats_array[i] = 5
	return cats_array



def marks2num(marks, cats_array):
	length = len(cats_array)
	numfea = []
	for i in range(len(cats_array)):
		if cats_array[i] == 0:
			tmp = num2id(marks[i])
			numfea=numfea + tmp
		elif cats_array[i] == 2:
			continue
		elif cats_array[i] == 1:
			tmp = mark2id(marks[i], phoneme_list)
			numfea=numfea + tmp
		elif cats_array[i] == 3:
			tmp = tune2id(marks[i], tune_list)
			numfea=numfea + tmp	
		elif cats_array[i] == 4:
			tmp = boundary2id(marks[i], boundary_list)
			numfea=numfea + tmp
		elif cats_array[i] == 5:
			tmp = zero2id(marks[i], zero_list)
			numfea=numfea + tmp
		else:
			print "marks error"
			break
	return numfea

def zero2id(mark, zero_list):
	if mark == 'x' or mark == '1':
		id_array = [0]
	elif mark == '2':
		id_array =[1]
	else:
		print "zero error"
		exit()
	return id_array

# def tune2id(mark, tune_list):
# 	if mark =='x':
# 		id_array = [0,0,0]
# 	elif mark == '1' :
# 		id_array = [0,0,1]
# 	elif mark == '2':
# 		id_array = [0,1,0]
# 	elif mark == '3':
# 		id_array = [0,1,1]
# 	elif mark == '4':
# 		id_array = [1,0,0]
# 	elif mark == '5':
# 		id_array = [1,0,1]
# 	elif mark == 'xx':
# 		id_array = [1,1,1]
# 	else:
# 		print "tune error"
# 		exit()
# 	return id_array


def tune2id(mark, tune_list):
	if mark =='x':
		id_array = [0,0,0,0,0]
	elif mark == '1' :
		id_array = [0,0,0,0,1]
	elif mark == '2':
		id_array = [0,0,0,1,0]
	elif mark == '3':
		id_array = [0,0,1,0,0]
	elif mark == '4':
		id_array = [0,1,0,0,0]
	elif mark == '5':
		id_array = [1,0,0,0,0]
	elif mark == 'xx':
		id_array = [1,1,1,1,1]
	else:
		print "tune error"
		exit()
	return id_array

def boundary2id(mark, boundary_list):
	if mark == '4':
		id_array = [1, 1, 1, 1]
	elif mark == '3':
		id_array = [1, 1, 1, 0]
	elif mark == '1':
		id_array = [1, 1, 0, 0]
	elif mark == '0':
		id_array = [1, 0, 0, 0]
	elif mark == 'x':
		id_array = [0, 0, 0, 0]
	else:
		print "bound error"
		exit()
	return id_array




def mark2id(mark, mark_list):
	if 'x' in mark_list:
		mark_list.remove('x')
	id_array = [0 for i in range(len(mark_list))]
	if mark != 'x':
		mark_index = mark_list.index(mark)
		id_array[mark_index] = 1
	return id_array
		
def num2id(mark):
	if mark == 'x':
		id = [0]
	else:
		id = [int(mark)]
	return id

def print_marks(lab_path, nums, list_path):
	marks = [[] for i in range(nums)]
	lab_files = os.listdir(lab_path)
	for file in lab_files:
		f = open(lab_path+"/"+file)
		iter_f = iter(f)
		for line in iter_f:
			line=line.split()[2]
			tmp = ext_mark(line)
			for i in range(nums):
				if not tmp[i] in marks[i]:
					marks[i].append(tmp[i])
	for i in range(nums):
		marks[i].sort()
		#print marks[i]
		#marks[i] = [j for j in set(marks[i])]
		f_name = list_path + '/list' + str(i)
		f=open(f_name, 'w+')
		for j in range(len(marks[i])):
			f.write(marks[i][j]+'\n')
		f.close()

def ext_mark(line):
	pos = 0
	res = []
	temp = ''
	i = 0

	while i<len(line):
		len_del = len(delimiter[pos])
		if pos >= len(delimiter) or line[i: i+len_del] != delimiter[pos]:
			temp += line[i]
			i += 1 
   		else:
   			pos += 1
	   		res.append(temp)
	   		temp = ''
	   		i += len_del
	res.append(temp)
	return res


def print_numfea(lab_path, nums, numfea_path, cats_array):
	marks = [[] for i in range(nums)]
	lab_files = os.listdir(lab_path)
	for file in lab_files:
		f = open(lab_path+"/"+file)
		iter_f = iter(f)

		fw = open(numfea_path+"/"+file, 'w')
		for line in iter_f:
			line=line.split()[2]
			tmp = ext_mark(line)
			assert len(tmp) == nums, "marks size error"
			nfea = marks2num(tmp, cats_array)
			#fw.writelines(str(nfea))
			print >> fw, nfea
		fw.close()
	
		

def acous_fea(cmp_path, par_path, mgc_dim):
	
	buf = open('run_par2vec.sh', 'w')  
	print >>buf, "cmp_path=%s" %(cmp_path)
	print >>buf, "par_path=%s" %(par_path)
	print >>buf, "mgc_dim=%d" %(mgc_dim) 

	print >> buf, "matlab -nojvm -nosplash -nodesktop -r \"cmp_dir='$cmp_path', par_dir='$par_path',mgc_dim=$mgc_dim;par2vec;quit\""
	buf.close()

	os.system("bash run_par2vec.sh")

def lab_align(align_path, nfea_path, align_nfea_path):
	
	for file in os.listdir(align_path):
		base = file.split('.')[0]
		print "analysizing %s" %(base)

		nfea_file = nfea_path +'/'+ base +'.lab'
		align_file = align_path + '/' + base + '.lab'
		align_nfea_file = align_nfea_path + '/' + base + '.input'
		#for line in open(nfea_file):
		#	nfea = line.lstrip('[').split(']')[0]
		with open(nfea_file) as f:
			nfeas = f.readlines()

		buf = open(align_nfea_file, 'w')
		count1 = 0 # control state num
		count2 = 0
		for line in open(align_file):

			nfea_line = nfeas[count1]
			nfea_line = nfea_line.lstrip('[').split(']')[0]

			line = line.lstrip()
			line = line.split(' ')
			start = int(line[0])/ftime
			end = int(line[1])/ftime
			fra_num = end - start # frame number
			state = line[2].split('[')[1][0]

			for i in range(fra_num):
				output = nfea_line+', '+state+', '+str(i+1) 
				output = output.replace(',', '')
				print >>buf, output

			count2 = count2 + 1
			if count2%5 == 0:
				count1 = count1 + 1
		buf.close()
		assert count2/5 == len(nfeas), "align error"

def rand_val(align_nfea_path, par_path, val_input, val_target):
	if not os.path.exists(val_input):
		os.makedirs(val_input)
	else:
		shutil.rmtree(val_input)
		os.makedirs(val_input)
	if not os.path.exists(val_target):
		os.makedirs(val_target)
	else:
		shutil.rmtree(val_target)
		os.makedirs(val_target)
	all_files = os.listdir(align_nfea_path)
	random.shuffle(all_files)#[0:100]
	val_files = all_files[0:100]
	#print val_files
	for file in val_files:
		base = file.split('.')[0]
		print "validation file: %s" %(base)
		src_input = align_nfea_path + '/' + base + '.input'
		src_target = par_path + '/' + base + '.target'
		#dest_input = val_input + '/' + base + '.input'
		#dest_target =  val_target + '/' + base + '.target'
		dest_input = val_input
		dest_target =  val_target
		shutil.copy(src_input, dest_input)
		shutil.copy(src_target, dest_target)

def select_val(align_nfea_path, par_path, val_input, val_target, val_scp):
	if not os.path.exists(val_input):
		os.makedirs(val_input)
	if not os.path.exists(val_target):
		os.makedirs(val_target)
	with open(val_scp) as f:
		content = f.readlines()
	content = [x.strip('\n') for x in content] 
	for file in val_files:
		print "validation file: %s" %(file)
		src_input = align_nfea_path + '/' + base + '.input'
		src_target = par_path + '/' + base + '.target'
		dest_input = val_input
		dest_target =  val_target
		shutil.move(src_input, dest_input)
		shutil.move(src_target, dest_target)


def train_nc_write(train_input, train_target, mean_dir, nc_file, switch, dim_input, dim_target):
	buf_name = "%s_nc_write.sh" %(switch)
	buf = open(buf_name, 'w')  
	print >>buf, "train_input=%s" %(train_input)
	print >>buf, "train_target=%s" %(train_target)
	print >>buf, "mean_dir=%s" %(mean_dir) 
	print >>buf, "nc_file=%s" %(nc_file) 
	print >>buf, "switch=%s" %(switch) 
	print >>buf, "dim_input=%s" %(dim_input) 
	print >>buf, "dim_target=%s" %(dim_target)
	print >> buf, "matlab -nojvm -nosplash -nodesktop -r \"file_input='$train_input', file_target='$train_target', \
	mean_dir='$mean_dir', nc_filename='$nc_file', dim_input=$dim_input,dim_target=$dim_target; \
	%s_nc_write;quit\"" %(switch)
	
	buf.close()

	os.system("bash %s" %(buf_name))

#def val_nc_write():

#def test_nc_write():
def del_file(tmp1, tmp2):
	list1 = os.listdir(tmp1)
	list1 = [l.strip('.input') for l in list1]
	list2 = os.listdir(tmp2)
	list2 = [l.strip('.target') for l in list2]
	l1 = [l for l in list1 if not l in list2]
	l2 = [l for l in list2 if not l in list1]
	#print l1
	#print l2
	for i in l1:
		os.remove(tmp1 + '/' + i + '.input')
	for i in l2:
		os.remove(tmp2 + '/' + i + '.target')

def gv_gen(dim_target, par_path, gv_path):
	buf = open('run_gv_gen.sh', 'w')  
	print >>buf, "gv_path=%s" %(gv_path)
	print >>buf, "par_path=%s" %(par_path)
	print >>buf, "dim_target=%d" %(dim_target) 
	print >> buf, "matlab -nojvm -nosplash -nodesktop -r \"gv_path='$gv_path', file_target='$par_path',dim_target=$dim_target;gen_gv;quit\""
	buf.close()

	os.system("bash run_gv_gen.sh")

def train_net(net_path, nc_path, stru, nodes, learning_rate, momentum, parallel_sequences, dim_input, dim_target):
	
	jsn_path = net_path + '/jsn'
	if not os.path.exists(jsn_path):
		os.makedirs(jsn_path)
	jsn_file = jsn_path + '/'+ stru + '_' +nodes + '.jsn'
	gen_jsn(jsn_file, stru, nodes, dim_input, dim_target)

	cfg_path = net_path +  '/cfg'
	if not os.path.exists(cfg_path):
		os.makedirs(cfg_path)
	cfg_file = cfg_path + '/' + stru + '_' +nodes + '.cfg'
	gen_cfg_train(cfg_file, stru, nodes, net_path, nc_path, learning_rate, momentum, parallel_sequences)

	

	sh_path = net_path + '/sh'
	if not os.path.exists(sh_path):
		os.makedirs(sh_path)
	train_file = sh_path + '/train_' + stru + '_' + nodes + '.sh'
	gen_sh_train(train_file, cfg_file)

	log_path = net_path + '/log'
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	log_file = log_path + '/' + stru + '_' + nodes + '.log'
	os.system("chmod +x %s" %(train_file))
	os.system("nohup bash %s > %s &" %(train_file, log_file))

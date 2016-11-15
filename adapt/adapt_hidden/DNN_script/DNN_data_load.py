import sys
sys.path.insert(0, "../../python/")
import numpy as np
import logging
from config_util import parse_args
import os
import scipy.io as scio
import glob

from joblib import Parallel, delayed
import multiprocessing

logging.basicConfig(level=logging.DEBUG)

num_cores = multiprocessing.cpu_count()

def norm_data(feats, tgts, mean_file):
	 (input_mean, input_std, output_mean, output_std) = ReadMat(mean_file)
	 loaded_feats = np.divide((feats - input_mean),input_std)
	 loaded_tgts = np.divide((tgts - output_mean), output_std)
	 return loaded_feats, loaded_tgts

def get_data(input_name, file_target):
	f_path = input_name.strip(".input").split("/")
	cmu = [s for s in f_path if "cmu" in s][0]
	utt_id =[s for s in f_path if "arctic" in s][0]
	output_name = input_name.split("INPUT")[0] + "TARGET" + "/"+utt_id+".target"
	feats = np.loadtxt(input_name).tolist()
	tgs = np.loadtxt(output_name).tolist()
	utt_id = cmu+"_"+utt_id
	results = [feats, tgs, utt_id, len(feats)]
	return results

def read_data(train_input, train_output, xdim, ydim):

	input_path_all = train_input.split(',')
	target_path_all = train_output.split(',')
	len_input_path_all = len(input_path_all)
	assert len(input_path_all) == len(target_path_all), 'input_path_len != target_path_len'

	features = []
	labels = []
	utt_ids = []
	num_frame = []
	for i in range(len_input_path_all):
		file_input = glob.glob(input_path_all[i]+"/*.input")
		file_target = glob.glob(target_path_all[i]+"/*.target")
		results = Parallel(n_jobs=num_cores)(delayed(get_data) (j, file_target) for j in file_input)

		print len(results)
	   
	  #  feats = Parallel(n_jobs=num_cores)(delayed(np.loadtxt)(j) for j in file_input)

	 #   file_target = glob.glob(target_path_all[i]+"/*.target")
		for j in range(len(results)):
			features.extend(results[j][0])
			labels.extend(results[j][1])
			utt_ids.append(results[j][2])
			num_frame.append(results[j][3])
		sum_num_frame = sum(num_frame)
		sc = np.zeros((sum_num_frame, len_input_path_all))
		sc[:,i] = 1
	return np.asarray(features), np.asarray(labels), utt_ids, num_frame, sc


def read_data_train(train_sets, xdim, ydim):
	sys.stderr.write('Loading data into memory...\n')
	features = []
	labels = []
	features = np.empty(shape=[0, xdim])
	labels = np.empty(shape=[0,ydim])
	utt_ids =[]
	utt_frame = []
	seq_len_tot = 0
	while True:
		(feats, tgs, utt_id) = train_sets.load_next_seq()
		if utt_id is None:
			break
		if feats.shape[0] == 0:
			continue

		seq_len_tot += feats.shape[0]
		features=np.append(features, feats, axis=0)
		labels=np.append(labels, tgs, axis=0)
		utt_ids.append(utt_id)
		utt_frame.append(feats.shape[0])
		utt_ids_frame = [utt_id, utt_frame]

	sys.stderr.write('	%d utterances loaded...\n' % len(utt_ids))
	sys.stderr.write('	avg-sequence-len = %.0f\n' % (seq_len_tot/len(utt_ids)))
	sys.stderr.write('	total frame number = %d \n' % seq_len_tot)	
	return features, labels, utt_ids_frame


def ReadMat(path_mean):
	input_mean= scio.loadmat(path_mean+"/input_mean.mat").values()[0]
	input_std= scio.loadmat(path_mean+"/input_std.mat").values()[0]
	output_mean= scio.loadmat(path_mean+"/output_mean.mat").values()[0]
	output_std= scio.loadmat(path_mean+"/output_std.mat").values()[0]
	return input_mean, input_std, output_mean, output_std

class my_DataRead(object):

	def __init__(self, dataset_args):
		self.input_mean = None
		self.input_std = None
		self.output_mean = None
		self.output_std = None
		self.input_files = []
		self.target_files = []
		self.len_input_files=[]
		self.fileIndex =[]

		if 'mean_file' in dataset_args.keys():
			(self.input_mean, self.input_std, self.output_mean, self.output_std) = ReadMat(dataset_args["mean_file"])

		input_path_all = dataset_args["input_file"].split(',')
		#print input_path_all
		#exit ()
		self.target_path_all = dataset_args["target_file"].split(',')
		self.len_input_path_all = len(input_path_all)

		assert len(input_path_all) == len(self.target_path_all), 'input_path_len != target_path_len'
		for i in range(len(input_path_all)):
			self.input_files.extend(glob.glob(input_path_all[i]+"/*.input"))
			self.target_files.extend(glob.glob(self.target_path_all[i]+"/*.target"))

		self.len_input_files=len(self.input_files)
		assert len(self.input_files)==len(self.target_files), "files number error "

		self.initialize_read()

	def initialize_read(self):
		self.fileIndex=0

	def load_next_seq(self):
		self.fileIndex+= 1
		if (self.fileIndex > self.len_input_files):
		#if (self.fileIndex[i] > 10):
			self.utt_id = None;
			loaded_feats = None;
			loaded_tgts = None;
			print ("read files %d over" % i)
		else:

			self.utt_id = os.path.basename(self.input_files[self.fileIndex-1]).strip(".input")
			feats = np.loadtxt(self.input_files[self.fileIndex-1])
			tsg_path = self.input_files[self.fileIndex-1].split("INPUT")[0] + "TARGET" 
			tgts = np.loadtxt(tsg_path+"/"+self.utt_id+".target")
			assert feats.shape[0] == tgts.shape[0], "frames number error"
			loaded_feats = np.divide((feats - self.input_mean),self.input_std)
			loaded_tgts = np.divide((tgts - self.output_mean), self.output_std)
			loaded_feats = feats
			loaded_tgts = tgts
	   # print self.fileIndex
		return loaded_feats, loaded_tgts, self.utt_id


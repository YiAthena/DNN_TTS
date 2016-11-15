#(feats, tgs, utt_id) = self.train_sets.load_next_seq()
import os
import sys
import glob
import numpy as np
import scipy.io as scio

class my_DataRead(object):
    def __init__(self, dataset_args, utt_num=None):
        
        self.max_feat = np.loadtxt('max_feat.txt')
        self.max_label = np.loadtxt('max_label.txt')
        self.min_label = np.loadtxt('min_label.txt')
        
        self.initialize_read()

        self.input_path = dataset_args["input_file"]
        self.target_path = dataset_args["target_file"]
        self.input_files = glob.glob(self.input_path+"/*.input")
        self.input_files.sort()
        self.target_files = glob.glob(self.target_path+"/*.target")
        self.target_files.sort()
        print (len(self.input_files))
        print (len(self.target_files))
        assert len(self.input_files)==len(self.target_files), "files number error"
        self.len_input_files = len(self.input_files)
        if utt_num == None:
            self.utt_num = self.len_input_files
        else:
            self.utt_num = utt_num

    def initialize_read(self):
        self.fileIndex = 0

    def load_next_seq(self):
        self.fileIndex += 1
            #if (self.fileIndex > self.len_input_files):
        if (self.fileIndex > self.utt_num):
            self.utt_id = None;
            feats = None;
            tgts = None;
            self.fileIndex = 0
            print "read files over"
        else:
            self.utt_id = os.path.basename(self.input_files[self.fileIndex-1]).strip(".input");
            feats = np.loadtxt(self.input_files[self.fileIndex-1])
            tgts = np.loadtxt(self.target_path+"/"+self.utt_id+".target")
            assert feats.shape[0] == tgts.shape[0], "frames number error"

            feats = np.divide(feats, self.max_feat) - 0.5
            tgts = np.divide((tgts-self.min_label),(self.max_label-self.min_label))
            
       # print self.fileIndex
        return feats, tgts, self.utt_id





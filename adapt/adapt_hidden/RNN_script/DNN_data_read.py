#(feats, tgs, utt_id) = self.train_sets.load_next_seq()
import os
import sys
import glob
import numpy as np
import scipy.io as scio

def read_data_train(train_sets, utt_num = None):
    sys.stderr.write('Loading data into memory...\n')
    features = []
    labels = []
    features = np.empty(shape=[0, 309])
    labels = np.empty(shape=[0, 202])
    utt_ids = []
    fr = []
    seq_len_tot = 0
    count=0
    if utt_num == None:
        num = train_sets.len_input_files
    else:
        num = utt_num
    for i in range(num):
        (feats, tgs, utt_id) = train_sets.load_next_seq()
        if utt_id is None:
            break
        if feats.shape[0] == 0:
            continue

        seq_len_tot += feats.shape[0]
        features=np.append(features, feats, axis=0)
        labels=np.append(labels, tgs, axis=0)
        utt_ids.append(utt_id)
        fr.append(feats.shape[0])
    sys.stderr.write('    %d utterances loaded...\n' % len(utt_ids))
    sys.stderr.write('    avg-sequence-len = %.0f\n' % (seq_len_tot/len(utt_ids)))
    sys.stderr.write('    total frame number = %d \n' % seq_len_tot)

    return features, labels, utt_ids, fr

class my_DataRead(object):
    def __init__(self, dataset_args):
        
        
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

    def initialize_read(self):
        self.fileIndex = 0

    def load_next_seq(self):
        self.fileIndex += 1
        if (self.fileIndex > self.len_input_files):
        #if (self.fileIndex > 10):
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
        print self.fileIndex    
        return feats, tgts, self.utt_id





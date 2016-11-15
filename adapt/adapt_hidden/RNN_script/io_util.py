import mxnet as mx
import numpy as np
import sys

class TruncatedSentenceIter(mx.io.DataIter):

	def __init__(self, train_sets, batch_size, init_states, truncate_len, delay,
				 feat_dim, label_dim, data_name, label_name,
				 do_shuffling, pad_zeros):

		self.train_sets = train_sets
		self.data_name = data_name
		self.label_name = label_name

		self.feat_dim = feat_dim
		self.label_dim = label_dim

		self.batch_size = batch_size
		self.truncate_len = truncate_len
		self.delay = delay


		self.do_shuffling = do_shuffling
		self.pad_zeros = pad_zeros

		self.data = [mx.nd.zeros((batch_size, truncate_len, feat_dim))]
		self.label = [mx.nd.zeros((batch_size, truncate_len, label_dim))]

		self.init_state_names = [x[0] for x in init_states]
		self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

		self.provide_data = [(data_name, self.data[0].shape)] + init_states
		self.provide_label = [(label_name, self.label[0].shape)]

		self._load_data()
		self._make_data_plan()

	def _load_data(self):
		sys.stderr.write('Loading data into memory...\n')
		self.features = []
		self.labels = []
		self.utt_ids = []

		seq_len_tot = 0.0
		for i in range (len(self.train_sets)):
			self.train_sets[i].initialize_read()
			while True:
				(feats, tgs, utt_id) = self.train_sets[i].load_next_seq()
				if utt_id is None:
					break
				if tgs is None and self.has_label:
					continue
				if feats.shape[0] == 0:
					continue

				tgs[self.delay:] = tgs[:-self.delay]
				tgs[:self.delay] = tgs[0]

				self.features.append(feats)
				self.labels.append(tgs+1)
				self.utt_ids.append(utt_id)
				seq_len_tot += feats.shape[0]

			sys.stderr.write('	%d frames loaded...\n' % len(self.features))
			sys.stderr.write('	%d utterances loaded...\n' % len(self.utt_ids))
			sys.stderr.write('	avg-sequence-len = %.0f\n' % (seq_len_tot/len(self.utt_ids)))

	def _make_data_plan(self):
		if self.do_shuffling:
			# TODO: should we group utterances of similar length together?
			self._data_plan = np.random.permutation(len(self.features))
		else:
			# we might not want to do shuffling for testing for example
			self._data_plan = np.arange(len(self.features))

	def __iter__(self):
		assert len(self._data_plan) >= self.batch_size, \
			"Total number of sentences smaller than batch size, consider using smaller batch size"
		utt_idx = self._data_plan[:self.batch_size]
		utt_inside_idx = [0] * self.batch_size

		next_utt_idx = self.batch_size
		is_pad = [False] * self.batch_size
		pad = 0

		np_data_buffer = np.zeros((self.batch_size, self.truncate_len, self.feat_dim))
		np_label_buffer = np.zeros((self.batch_size, self.truncate_len, self.label_dim))
		utt_id_buffer = [None] * self.batch_size

		data_names = [self.data_name] + self.init_state_names
		label_names = [self.label_name]

		# reset states
		for state in self.init_state_arrays:
			state[:] = 0.1

		while True:
			effective_sample_count = self.batch_size * self.truncate_len
			for i, idx in enumerate(utt_idx):
				fea_utt = self.features[idx]
				if utt_inside_idx[i] >= fea_utt.shape[0]:
					# we have consumed this sentence

					# reset the states
					for state in self.init_state_arrays:
						state[i:i+1] = 0.1
					# load new sentence
					if is_pad[i]:
						# I am already a padded sentence, just rewind to the
						# beginning of the sentece
						utt_inside_idx[i] = 0
					elif next_utt_idx >= len(self.features):
						# we consumed the whole dataset, simply repeat this sentence
						# and set pad
						pad += 1
						is_pad[i] = True
						utt_inside_idx[i] = 0
					else:
						# move to the next sentence
						utt_idx[i] = self._data_plan[next_utt_idx]
						idx = utt_idx[i]
						fea_utt = self.features[idx]
						utt_inside_idx[i] = 0
						next_utt_idx += 1

				if is_pad[i] and self.pad_zeros:
					np_data_buffer[i] = 0
					np_label_buffer[i] = 0
					effective_sample_count -= self.truncate_len
				else:
					idx_take = slice(utt_inside_idx[i],
									 min(utt_inside_idx[i]+self.truncate_len,
										 fea_utt.shape[0]))
					n_take = idx_take.stop - idx_take.start
					np_data_buffer[i][:n_take] = fea_utt[idx_take]
					np_label_buffer[i][:n_take] = self.labels[idx][idx_take]
					if n_take < self.truncate_len:
						np_data_buffer[i][n_take:] = 0
						np_label_buffer[i][n_take:] = 0
						effective_sample_count -= self.truncate_len - n_take

					utt_inside_idx[i] += n_take

				utt_id_buffer[i] = self.utt_ids[idx]

			if pad == self.batch_size:
				# finished all the senteces
				break

			self.data[0][:] = np_data_buffer
			self.label[0][:] = np_label_buffer
			data_batch = SimpleBatch(data_names, self.data + self.init_state_arrays,
									 label_names, self.label, bucket_key=None,
									 utt_id=utt_id_buffer,
									 effective_sample_count=effective_sample_count)

			# Instead of using the 'pad' property, we use an array 'is_pad'. Because
			# our padded sentence could be in the middle of a batch. A sample is pad
			# if we are running out of the data set and they are just some previously
			# seen data to be filled for a whole batch. In prediction, those data
			# should be ignored
			data_batch.is_pad = is_pad

			yield data_batch

	def reset(self):
		self._make_data_plan()


class SimpleBatch(object):
	def __init__(self, data_names, data, label_names, label, bucket_key,
				 utt_id=None, utt_len=0, effective_sample_count=None):
		self.data = data
		self.label = label
		self.data_names = data_names
		self.label_names = label_names
		self.bucket_key = bucket_key
		self.utt_id = utt_id
		self.utt_len = utt_len
		self.effective_sample_count = effective_sample_count

		self.pad = 0
		self.index = None  # TODO: what is index?

	@property
	def provide_data(self):
		return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

	@property
	def provide_label(self):
		return [(n, x.shape) for n, x in zip(self.label_names, self.label)]








import numpy as np
import random
import yaml

class FeatureVector(object):

	def __init__(self, data):
		self._features = data
		self._ID = data[0]
		self._date = [d for d in data[1:4]]
		self._time = [d for d in data[4:7]]

	def __repr__(self):
		return 'ID: {0}'.format(self.ID)

	def __getitem__(self, index):
		return self.features[index]

	@property
	def features(self):
		return self._features

	@features.setter
	def features(self, value):
		self.features = value

	@property
	def date(self):
		 return self._date

	@date.setter
	def date(self, value):
		self.date = value

	@property
	def time(self):
	    return self._time

	@time.setter
	def time(self, value):
	    self._time = value

	@property
	def ID(self):
		return self._ID

	@ID.setter
	def ID(self, value):
		self.ID = value


class InfoMatrix(object):

	def __init__(self, data):
		self._vecs = []
		for d in data:
			self._vecs.append(FeatureVector(d))

	def __iter__(self):
		return iter(self.vecs)

	@property
	def vecs(self):
		return self._vecs


class DataStruct(object):

	def __init__(self, data, labels):
		with open('definitions.yml', 'r') as f:
			self._conf = yaml.load(f)
		self._data = []
		self._labels = []
		self._raw_data = data
		self._raw_labels = labels
		self._test_size = int(self._conf['data_size'] * self._conf['test_percentage'])
		self._max_values = np.maximum(np.max(data, (0, 1)), np.max(labels, 0))
		self._min_values = np.minimum(np.min(data, (0, 1)), np.min(labels, 0))

		data, labels = self._normalize(data, labels)


		for data_piece, label in zip(data, labels):
			self._data.append([FeatureVector(feature_data) for feature_data in data_piece])
			self._labels.append(FeatureVector(label))

			# self._raw_data.append(data_piece)
			# self._raw_labels.append(label)

		self._shuffle()

	@property
	def data(self):
	    return self._data

	@property
	def labels(self):
		return self._labels

	@property
	def raw_data(self):
		return self._raw_data

	@property
	def raw_labels(self):
		return self._raw_labels

	@property
	def conf(self):
		return self._conf

	@property
	def max_values(self):
	    return self._max_values

	@property
	def min_values(self):
	    return self._min_values

	@property
	def test_size(self):
		return self._test_size

	def get_batch(self, batch_number):
		batch_slice = slice(self.test_size + batch_number * self.conf['batch_size'],
							self.test_size + (batch_number + 1) * self.conf['batch_size'])
		# return np.concatenate(self.raw_data[batch_slice]), np.concatenate(self.raw_labels[batch_slice])
		return self.raw_data[batch_slice], self.raw_labels[batch_slice]

	def get_test(self):
		# return np.concatenate(self.raw_data[:self.test_size]), np.concatenate(self.raw_labels[:self.test_size])
		return self.raw_data[:self.test_size], self.raw_labels[:self.test_size]

	def denormalize(self):
		pass

	def _normalize(self, data, labels):
		for feature_num in range(self.conf['number_features']):
			data[:, :, feature_num] = (data[:, :, feature_num] - self.min_values[feature_num]) / \
									  (self.max_values[feature_num] - self.min_values[feature_num])
			labels[:, feature_num] = (labels[:, feature_num] - self.min_values[feature_num]) / \
									 (self.max_values[feature_num] - self.min_values[feature_num])
		return data, labels

	def _shuffle(self):
		random_seed = random.random()
		random.shuffle(self._data, lambda: random_seed)
		random.shuffle(self._labels, lambda: random_seed)

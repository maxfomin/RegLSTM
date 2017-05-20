import numpy as np
import random

class FeatureVector(object):

	def __init__(self, data):
		self._features = data
		self._ID = data[0]
		self._date = [d for d in data[1:4]]
		self._time = [d for d in data[4:7]]

	def __repr__(self):
		return 'ID: {0}'.format(self.ID)

	@property
	def features(self):
		return self._data

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
		self._data = []
		self._labels = []
		self._test_size = int(conf['data_size'] * conf['test_percentage'])
		for data_piece, label in zip(data, labels):
			self._data.append(InfoMatrix(data_piece))
			self._labels.append(FeatureVector(label))

		random_seed = random.random()
		random.shuffle(self._data, lambda: random_seed)
		random.shuffle(self._labels, lambda: random_seed)
		self.normalize()

	@property
	def data(self):
	    return self._data

	@property
	def labels(self):
		return self._labels

	@property
	def test_size(self):
		return self._test_size

	def shuffle(self):
		pass

	def get_batch(self):
		pass

	def get_test(self):
		return self.data[:self.test_size], self.labels[:self.test_size]

	def normalize(self):
		pass

	def denormalize(self):
		pass

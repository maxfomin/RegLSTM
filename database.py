import numpy as np

class FeatureVector(object):

	def __init__(self, data):
		self._data = data
		self._ID = data[0]
		self._date = [d for d in data[1:4]]
		self._time = [d for d in data[4:7]]

	def __repr__(self):
		return 'ID: {0}'.format(self.ID)

	@property
	def data(self):
		return self._data

	@data.setter
	def data(self, value):
		self.data = value

	@property
	def date(self):
		self._date

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
		for data_piece, label in zip(data, labels):
			self._data.append(InfoMatrix(data_piece))
			self._labels.append(InfoMatrix(label))

	def __iter__(self):
		return iter(zip(self.data, self.labels))

	@property
	def data(self):
	    self._data

	@property
	def labels(self):
		return self._labels

	def shuffle(self):
		pass

	def get_batch(self):
		pass

	def tensor_output(self):
		pass

	def normalize(self):
		pass

	def denormalize(self):
		pass

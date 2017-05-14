import numpy as np
conf = {
	'information_length': 3,
	'number_of_features': 20
}

class FeatureVector:

	def __init__(self, data, ID):
		if len(data) != conf['number_of_features']:
			raise Exception('Data for the feature vector is not in the correct length')
		if ID is None:
			raise Exception('ID has to be supplied for the feature vector')

		self._data = data
		self._ID = ID

	def __repr__(self):
		return 'ID: {0}'.format(self.ID)

	@property
	def data(self):
		return self._data

	@data.setter
	def data(self, value):
		self.data = value

	@property
	def ID(self):
		return self._ID

	@ID.setter
	def ID(self, value):
		self.ID = value

class InfoMatrix:

	def __init__(self, data):
		if len(data) != conf['information_length']:
			raise Exception('Information length has to be {0}'.format(conf['information_length']))

		self._vecs = data

	def __iter__(self):
		return iter(self.vecs)

	@property
	def vecs(self):
		return self._vecs

	@vecs.setter
	def vecs(self, value):
		self.vecs = value

	def load_matrix(self):
		pass

	def generate_matrix(self):
		pass

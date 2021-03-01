from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.data_utils import Sequence
import scipy.stats

class MixupImageDataGenerator():
	def __init__(self, gen, directory, batch_size, img_height, img_width, distr, params, majority_vote = 0):
		
		self.batch_index = 0
		self.batch_size = batch_size
		self.params = params
		self.distr = distr
		self.shape = (img_height, img_width)
		self.majority_vote = majority_vote
		self.gen1 = gen.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True)
														
		self.gen2 = gen.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True)
														
														
		self.n = self.gen1.samples
		
	def reset_index(self):
		self.gen1._set_index_array()
		self.gen2._set_index_array()
	
	def __len__(self):
		return (self.n + self.batch_size - 1) // self.batch_size
	
	def steps_per_epoch(self):
		return self.n // self.batch_size
		
	def __next__(self):
		if self.batch_index == 0:
			self.reset_index()
	
	
		current_index = (self.batch_index * self.batch_size) % self.n
		if self.n > current_index + self.batch_size:
			self.batch_index += 1
		else:
			self.batch_index = 0
			
	
		# Get a pair of inputs and outputs from two iterators.
		X1, y1 = self.gen1.next()
		X2, y2 = self.gen2.next()
		
				# random sample the lambda value from beta distribution.
		if self.distr == "beta":
			l = np.random.beta(self.params[0], self.params[1], len(X1))
		if self.distr == "trunc_norm":
			l = scipy.stats.truncnorm.rvs((0-self.params[0])/self.params[1],(1-self.params[0])/self.params[1],loc=self.params[0],scale=self.params[1],size=len(X1))
			
		X_l = l.reshape(len(X1), 1, 1, 1)
		y_l = l.reshape(len(y1), 1)
		
		
		
		# Perform the mixup
		X = X1 * X_l + X2 * (1 - X_l)
		if self.majority_vote == 1:
			l[l > 0.5] = 1
			l[l < 0.5] = 0	
		y = y1 * y_l + y2 * (1 - y_l)
		
		return (X, y)
	
	def generate(self):
		while True:
			yield next(self)
			
			
			
		
		
		
		

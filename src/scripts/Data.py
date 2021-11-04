import os
import pandas as pd
import numpy as np
from sklearn import model_selection

class Data:
	def __init__(self, path):
		self.images = self.__extract_images(path)
		self.images.category, self.labels = self.images.category.factorize()
		self.training, self.test = None, None

	def train_test_split(self, test_size=0.15, shuffle=True, stratify=False):
		return model_selection.train_test_split(
			self.images,
			test_size=test_size,
			random_state=42,
			shuffle=shuffle,
			stratify=self.images.category if stratify else None
		)

	def count_labels(self, data, name):
		amount = data.category.value_counts().values
		
		print(f"{name}: {amount} {np.round(amount/len(data), 2)}")

	def image_generator(self):
		pass

	def __extract_images(self, path):
		images = []
		
		for category in os.listdir(path):
			for filename in os.listdir(path + category):
				images.append([path + category + "/" + filename, category])

		return pd.DataFrame(images, columns=["image", "category"])

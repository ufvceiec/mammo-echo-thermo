import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from keras import preprocessing
from .Misc import *

class Data:
	def __init__(self, path):
		self.images = self.__extract_images(path)
		self.images.category, self.labels = self.images.category.factorize()
		self.images.category = self.images.category.astype(str)
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
		train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
		test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)

		generator_properties = {
			"x_col": "image",
			"y_col": "category",
			"target_size": (215, 538),
			"color_mode": "rgb",
			"class_mode": "categorical"
		}

		train_generator = train_datagen.flow_from_dataframe(
			**generator_properties,

			dataframe=self.training,
			batch_size=25,
			shuffle=True,
			subset="training"
		)

		validation_generator = train_datagen.flow_from_dataframe(
			**generator_properties,

			dataframe=self.training,
			batch_size=25,
			shuffle=True,
			subset="validation"
		)

		test_generator = test_datagen.flow_from_dataframe(
			**generator_properties,

			dataframe=self.test,
			batch_size=1,
			shuffle=False
		)

		return train_generator, validation_generator, test_generator

	def show_images(self, generator, name):
		fig, ax = plt.subplots(nrows=3, ncols=1, constrained_layout=True)

		img, label = generator.next()

		for i in range(3):
			ax[i].imshow(img[i])
			ax[i].title.set_text(str(self.labels[np.argmax(label[i], axis=-1)]).title())
			ax[i].axis("off")

		fig.suptitle(name)
			
		plt.show()

	def __extract_images(self, path):
		images = []
		
		for category in os.listdir(path):
			for filename in os.listdir(path + category):
				images.append([path + category + "/" + filename, category])

		return pd.DataFrame(images, columns=["image", "category"])

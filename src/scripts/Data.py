import os
import pandas as pd
import numpy as np
import cv2 as cv
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

	def image_generator(self, shuffle=True):
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
			batch_size=10,
			shuffle=shuffle,
			subset="training"
		)

		validation_generator = train_datagen.flow_from_dataframe(
			**generator_properties,

			dataframe=self.training,
			batch_size=10,
			shuffle=shuffle,
			subset="validation"
		)

		test_generator = test_datagen.flow_from_dataframe(
			**generator_properties,

			dataframe=self.test,
			batch_size=1,
			shuffle=False
		)

		return train_generator, validation_generator, test_generator

	def detectColor(self, image, lower, upper):
		if tf.is_tensor(image):
			temp_image = image.numpy().copy()
		else:
			temp_image = image.copy()

		hsv_image = temp_image.copy()
		hsv_image = cv.cvtColor(hsv_image, cv.COLOR_RGB2HSV)
		mask = cv.inRange(hsv_image, lower, upper)

		result = temp_image.copy()
		result[np.where(mask == 0)] = 0
		
		return result

	def getImageTensor(self, images, lower, upper):
		results = []

		for img in images:
			results.append(np.expand_dims(self.detectColor(img, lower, upper), axis=0))

		return np.concatenate(results, axis=0)

	def show_images(self, generator, filters, name):
		generator.reset()

		img, label = generator.next()

		fig, axs = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
		fig.suptitle(name)

		for ax in axs:
			ax.remove()

		gridspec = axs[0].get_subplotspec().get_gridspec()
		subfigs = [fig.add_subfigure(gs) for gs in gridspec]

		for row, subfig in enumerate(subfigs):
			subfig.suptitle(str(self.labels[np.argmax(label[row], axis=-1)]).title())

			axs = subfig.subplots(nrows=1, ncols=4)

			for col, ax in enumerate(axs):
				ax.imshow(list(filters.values())[col](img)[row])
				ax.set_title(list(filters)[col].title())
				ax.axis("off")
				
				ax.plot()

	def __extract_images(self, path):
		images = []
		
		for category in os.listdir(path):
			for filename in os.listdir(path + category):
				images.append([path + category + "/" + filename, category])

		return pd.DataFrame(images, columns=["image", "category"])

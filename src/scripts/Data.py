import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import model_selection
from keras import preprocessing
from .Misc import *

class Data:
	def __init__(self, path, random_state=42):
		self.random_state = random_state
		self.path = path
		self.images = self.__extract_images(path) # The paths of the images and their categories are loaded into a dataframe
		self.images.category, self.labels = self.images.category.factorize() # Categories are converted to integers and the labels are stored in a dictionary
		self.images.category = self.images.category.astype(str)
		self.training, self.test = None, None

	# Split data into train and test
	def train_test_split(self, test_size=0.15, stratify=False):
		return model_selection.train_test_split(
			self.images,
			test_size=test_size,
			random_state=self.random_state,
			shuffle=True,
			stratify=self.images.category if stratify else None
		)

	# The number of each class in the dataset is counted
	def count_labels(self, data, name="Dataset"):
		amount = data.category.value_counts().values
		
		print(f"{name}: {amount} {np.round(amount/len(data), 2)}")

	# Function in charge of generating training and validation images
	def training_validation_generator(self, n_splits=5):
		kfold = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state) # Split the data into n_splits folds
		datagen, generator_properties = self.generator() # Generate the generator and its properties

		# For each fold, generate the training and validation images
		for training_index, validation_index in kfold.split(self.training):
			training_data = self.training.iloc[training_index]
			validation_data = self.training.iloc[validation_index]

			train_generator = datagen.flow_from_dataframe(
				**generator_properties,

				dataframe=training_data,
				batch_size=10,
				shuffle=False,
			)

			validation_generator = datagen.flow_from_dataframe(
				**generator_properties,

				dataframe=validation_data,
				batch_size=10,
				shuffle=False,
			)

			yield train_generator, validation_generator

	# Function in charge of generating test images
	def test_generator(self):
		datagen, generator_properties = self.generator() # Generate the generator and its properties

		test_generator = datagen.flow_from_dataframe(
			**generator_properties,

			dataframe=self.test,
			batch_size=1,
			shuffle=False
		)

		return test_generator

	def generator(self):
		datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)

		generator_properties = {
			"x_col": "image",
			"y_col": "category",
			"target_size": (215, 538),
			"color_mode": "rgb",
			"class_mode": "categorical"
		}

		return datagen, generator_properties

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

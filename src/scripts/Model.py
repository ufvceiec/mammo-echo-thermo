import matplotlib.pyplot as plt
from keras import models, layers, utils
from .Misc import *

class Model:
	def __init__(self):
		self.model = self.__create_model()

	def __create_model(self):
		model = models.Sequential()

		model.add(layers.Conv2D(32, (3, 3), input_shape=(215, 538, 3), activation="relu"))
		model.add(layers.MaxPooling2D(pool_size=(2, 2)))

		model.add(layers.Conv2D(32, (3, 3), activation="relu"))
		model.add(layers.GlobalAveragePooling2D())

		model.add(layers.Dropout(rate=0.4))
		model.add(layers.Dense(32, activation="relu"))
		model.add(layers.Dropout(rate=0.4))
		model.add(layers.Dense(2, activation="softmax"))

		return model

	def plot(self, name):
		computer.create_output_folder()

		utils.vis_utils.plot_model(self.model, to_file="./output/" + name + ".png", show_shapes=True, show_layer_names=True)

		plt.imshow(plt.imread("./output/" + name + ".png"))

		plt.show()

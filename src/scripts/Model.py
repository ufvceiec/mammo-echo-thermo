import matplotlib.pyplot as plt
from keras import callbacks, models, layers, utils, losses
from tensorflow.keras import optimizers
from .Misc import *

class Model:
	def __init__(self, name, summary=True, plot=True):
		self.model = self.__create_model()
		self.name = name

		if summary:
			self.model.summary()

		if plot:
			computer.create_output_folder()

			utils.vis_utils.plot_model(self.model, to_file=f"./output/{self.name}_model.png", show_shapes=True, show_layer_names=True)

			plt.imshow(plt.imread(f"./output/{self.name}_model.png"))
			plt.show()		

	def __create_model(self):
		model = models.Sequential()

		model.add(layers.Conv2D(32, (3, 3), input_shape=(215, 538, 3), activation="relu"))
		model.add(layers.MaxPooling2D(pool_size=(2, 2)))

		model.add(layers.Conv2D(32, (3, 3), activation="relu"))
		model.add(layers.GlobalAveragePooling2D())

		model.add(layers.Dropout(rate=0.4))
		model.add(layers.Dense(32, activation="relu"))
		model.add(layers.Dropout(rate=0.4))
		model.add(layers.Dense(1, activation="softmax"))

		return model

	def compile(self):
		self.model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(learning_rate=5*10e-5), metrics=["accuracy"])

		with open(f"./output/{self.name}_model.json", "w") as json_file:
			json_file.write(self.model.to_json())

	def fit(self, train_generator, validation_generator, epochs, plot=True):
		checkpoint = callbacks.ModelCheckpoint(f"./output/{self.name}_weigth.hdf5", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")

		history = self.model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[checkpoint])

		if plot:
			plt.style.use("ggplot")

			plt.figure()

			plt.plot(history.history["loss"], label="Training loss")
			plt.plot(history.history["val_loss"], label="Validation loss")
			plt.plot(history.history["accuracy"], label="Training accuracy")
			plt.plot(history.history["val_accuracy"], label="Validation accuracy")

			plt.title("Training Loss and Accuracy")
			plt.xlabel("Epoch #")
			plt.ylabel("Loss/Accuracy")
			plt.legend(loc="lower left")

			plt.show()

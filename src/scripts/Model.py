import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2 as cv
import pandas as pd
import tensorflow as tf
from keras import callbacks, models, layers, utils, losses, preprocessing
from sklearn import metrics
from tensorflow.keras import optimizers
from datetime import datetime
from .Misc import *

class Model:
	def __init__(self, path, filter):
		self.filter = filter
		self.__path = path
		self.model = self.__create_model() # The neural model is created

		computer.create_folder(self.__path) # The model folder is created
		
		utils.vis_utils.plot_model(self.model, to_file=f"{self.__path}/model.png", show_shapes=True, show_layer_names=True) # Saved model plot

	# TODO: Apply grid search to find the best parameters
	# Function in charge of creating the layers of the neural model
	def __create_model(self):
		model = models.Sequential()

		# TODO: Set image size parameters from an external function
		model.add(FilterLayer(filter=self.filter, name="filter_layer", input_shape=(215, 538, 3), trainable=False))

		model.add(layers.Conv2D(32, (3, 3), activation="relu"))
		model.add(layers.MaxPooling2D(pool_size=(2, 2)))

		model.add(layers.Conv2D(32, (3, 3), activation="relu"))
		model.add(layers.GlobalAveragePooling2D())

		model.add(layers.Dropout(rate=0.4))
		model.add(layers.Dense(32, activation="relu"))
		
		model.add(layers.Dropout(rate=0.4))
		model.add(layers.Dense(2, activation="softmax"))

		return model

	# Function in charge of compiling the model and saving it
	def compile(self):
		# TODO: Apply grid search to find the best parameters
		self.model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(learning_rate=5*10e-4), metrics=["accuracy"]) # Compile the model

		# Save the model architecture and weights
		with open(f"{self.__path}/model.json", "w") as json_file:
			json_file.write(self.model.to_json())

	# Function in charge of training the model
	def fit(self, train_generator, validation_generator, epochs, verbose=True):
		weighted_path = f"{self.__path}/weights_" + "{epoch:03d}" + ".hdf5"

		# Class in charge of following the training process
		class GetProgress(callbacks.Callback):
			def __init__(self, path):
				self.__path = path
				self.__best_accuracy = 0
				self.__best_epoch = 0
				self.__start_time = datetime.now().timestamp()

			# Function executed every time an epoch completes
			def on_epoch_end(self, epoch, logs=None):
				# The best model is saved as best_model.hdf5
				if logs["val_accuracy"] > self.__best_accuracy:
					self.__best_accuracy = logs["val_accuracy"]
					self.__best_epoch = epoch + 1
					computer.duplicate_file(weighted_path.format(epoch=self.__best_epoch), f"{self.__path}/best_model.hdf5")

				print(
					f"\rModel -> " +
					f"Epoch {epoch + 1}/{epochs} -> " +
					f"Accuracy (Validation): {round(logs['val_accuracy'], 2)} (Best: {round(self.__best_accuracy, 2)}) -> " +
					f"{weighted_path.format(epoch=self.__best_epoch)}"
				, end="")

			# Function executed when the training process is finished
			def on_train_end(self, logs=None):
				print(f"\nTraining finished in {datetime.now().timestamp() - self.__start_time} seconds \n")
				return super().on_train_end(logs=logs)

		# Callback stores the best model (best validation accuracy)
		checkpoint = callbacks.ModelCheckpoint(
			weighted_path,
			monitor="val_accuracy",
			verbose=1 if verbose else 0,
			save_best_only=True,
			mode="max"
		)

		get_progress = GetProgress(self.__path) # Callback to get the progress

		# Train the model
		history = self.model.fit(
			train_generator,
			epochs=epochs,
			verbose=1 if verbose else 0,
			validation_data=validation_generator,
			callbacks=[
				checkpoint,
				get_progress
			]
		)

		plt.style.use("ggplot")

		fig = plt.figure()

		plt.plot(history.history["loss"], label="Training loss")
		plt.plot(history.history["val_loss"], label="Validation loss")
		plt.plot(history.history["accuracy"], label="Training accuracy")
		plt.plot(history.history["val_accuracy"], label="Validation accuracy")

		plt.title("Training Loss and Accuracy")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")

		plt.savefig(f"{self.__path}/training_loss_and_accuracy.png") # Save the plot

		plt.close(fig)

	# Function in charge of loading the best model
	def load_model(self, path=None):
		self.model.load_weights(f"{self.__path}/best_model.hdf5" if path is None else path)

	# TODO: Save in a file the accuracy, specificity, sensitivity and precision of the model
	# Function in charge of evaluating a dataset
	def evaluate(self, predict, title, path=None):
		self.load_model(path) # Load the best model

		# Evaluate the model and get the confusion matrix
		predictions = np.argmax(self.model.predict(predict), axis=-1)
		cm = metrics.confusion_matrix(predict.classes, predictions)
		
		# Plot the confusion matrix
		metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(predict.labels)).plot(cmap=plt.cm.Blues, xticks_rotation=0)
		plt.savefig(f"{self.__path}/confusion_matrix_{title}.png")
		plt.show()

		print(metrics.classification_report(predict.classes, predictions, zero_division=0)) # Print the classification report

		TP = cm[1][1]
		TN = cm[0][0]
		FP = cm[0][1]
		FN = cm[1][0]

		accuracy = (float(TP + TN)/float(TP + TN + FP + FN))
		print("Accuracy:", round(accuracy, 4))

		specificity = (TN/float(TN + FP))
		print("Specificity:", round(specificity, 4))

		sensitivity = (TP/float(TP + FN))
		print("Sensitivity:", round(sensitivity, 4))

		precision = (TP/float(TP + FP))
		print("Precision:", round(precision, 4))

	# Function in charge of visualizing the model with the corresponding filter
	def visualize_heatmap(self, image):
		img = cv.cvtColor(cv.imread(image.image), cv.COLOR_BGR2RGB).astype("float32") * 1./255
		img = np.expand_dims(img, axis=0)

		heatmap = self.compute_heatmap(img)
		jet_heatmap, superimposed_img = self.get_heatmap(image.image, heatmap)

		fig, ax = plt.subplots(1, 3, figsize=(20, 8))

		ax[0].imshow(img[0])
		ax[0].set_title(os.path.basename(image.image))
		ax[1].imshow(preprocessing.image.array_to_img(jet_heatmap))
		ax[1].set_title(f"Real class: {image.category}")
		ax[2].imshow(superimposed_img)
		ax[2].set_title(f"Predicted class: {np.argmax(self.model.predict(img)[0])}")

		plt.show()

	# Function in charge of computing the heatmap
	def compute_heatmap(self, image):
		last_layer = self.model.get_layer(index=2)

		grad_model = models.Model(inputs=[self.model.input], outputs=[last_layer.output, self.model.output])

		with tf.GradientTape() as tape:
			model_output, last_layer = grad_model(image)

			grads = tape.gradient(last_layer[:, 1], model_output)
			pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

		heatmap = model_output[0] @ pooled_grads[..., tf.newaxis]
		heatmap = tf.squeeze(heatmap)
		heatmap = tf.maximum(heatmap, 0)/tf.math.reduce_max(heatmap)

		return heatmap.numpy()

	# Function in charge of getting the heatmap
	def get_heatmap(self, path, heatmap):
		img = preprocessing.image.load_img(path, color_mode="grayscale")
		img = preprocessing.image.img_to_array(img)

		jet = cm.get_cmap("jet")
		jet_colors = jet(np.arange(256))[:, :3]
		
		jet_heatmap = jet_colors[np.uint8(255 * heatmap)]
		jet_heatmap = preprocessing.image.array_to_img(jet_heatmap)
		jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
		jet_heatmap = preprocessing.image.img_to_array(jet_heatmap)

		superimposed_img = jet_heatmap * 0.4 + img
		superimposed_img = preprocessing.image.array_to_img(superimposed_img)

		return jet_heatmap, superimposed_img

# Custom layer in charge of applying a specific filter to the input
class FilterLayer(layers.Layer):
	def __init__(self, filter, name="filter_layer", **kwargs):
		self.filter = filter

		super(FilterLayer, self).__init__(name=name, **kwargs) # Initialize the layer

	# This function starts when the layer is called
	def call(self, image):
		shape = image.shape
		[image, ] = tf.py_function(self.filter, [image], [tf.float32]) # Apply the filter
		image.set_shape(shape)
		
		return image

	# This function starts when the layer is built
	def get_config(self):
		config = super(FilterLayer, self).get_config()
		config.update({"filter": self.filter}) # Add the filter to the config
		
		return config

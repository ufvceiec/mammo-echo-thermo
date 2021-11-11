import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2 as cv
import tensorflow as tf
from keras import callbacks, models, layers, utils, losses, preprocessing
from sklearn import metrics
from tensorflow.keras import optimizers
from .Misc import *

class Model:
	def __init__(self, name, filter, summary=True, plot=True):
		self.model = self.__create_model(filter)
		self.name = name

		if summary:
			self.model.summary()

		if plot:
			computer.create_output_folder()

			utils.vis_utils.plot_model(self.model, to_file=f"./output/{self.name}_model.png", show_shapes=True, show_layer_names=True)

			plt.imshow(plt.imread(f"./output/{self.name}_model.png"))
			plt.axis("off")

			plt.show()

	def __create_model(self, filter):
		model = models.Sequential()

		model.add(layers.Lambda(filter, input_shape=(215, 538, 3)))

		model.add(layers.Conv2D(32, (3, 3), activation="relu"))
		model.add(layers.MaxPooling2D(pool_size=(2, 2)))

		model.add(layers.Conv2D(32, (3, 3), activation="relu"))
		model.add(layers.GlobalAveragePooling2D())

		model.add(layers.Dropout(rate=0.4))
		model.add(layers.Dense(32, activation="relu"))
		model.add(layers.Dropout(rate=0.4))
		model.add(layers.Dense(2, activation="softmax"))

		return model

	def compile(self):
		self.model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(learning_rate=5*10e-4), metrics=["accuracy"])

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

	def load_model(self):
		self.model = models.load_model(f"./output/{self.name}_weigth.hdf5")

	def evaluate(self, predict, best_model=True):
		if best_model:
			self.load_model()

		predictions = np.argmax(self.model.predict(predict), axis=-1)
		cm = metrics.confusion_matrix(predict.classes, predictions)
		
		metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(predict.labels)).plot(cmap=plt.cm.Blues, xticks_rotation=0)
		plt.show()

		print(metrics.classification_report(predict.classes, predictions))

		TP = cm[1][1]
		TN = cm[0][0]
		FP = cm[0][1]
		FN = cm[1][0]

		accuracy = (float(TP + TN) / float(TP + TN + FP + FN))
		print("Accuracy:", round(accuracy, 4))

		specificity = (TN / float(TN + FP))
		print("Specificity:", round(specificity, 4))

		sensitivity = (TP / float(TP + FN))
		print("Sensitivity:", round(sensitivity, 4))

		precision = (TP / float(TP + FP))
		print("Precision:", round(precision, 4))

	def visualize_heatmap(self, image):
		img = cv.cvtColor(cv.imread(image.image), cv.COLOR_BGR2RGB).astype("float32") * 1./255
		img = np.expand_dims(img, axis=0)

		heatmap = self.__compute_heatmap(img)
		jet_heatmap, superimposed_img = self.__get_heatmap(image.image, heatmap)

		fig, ax = plt.subplots(1, 3, figsize=(20, 8))

		ax[0].imshow(img[0])
		ax[0].set_title(os.path.basename(image.image))
		ax[1].imshow(preprocessing.image.array_to_img(jet_heatmap))
		ax[1].set_title(f"Real class: {image.category}")
		ax[2].imshow(superimposed_img)
		ax[2].set_title(f"Predicted class: {np.argmax(self.model.predict(img)[0])}")

		plt.show()

	def __compute_heatmap(self, image):
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

	def __get_heatmap(self, path, heatmap):
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

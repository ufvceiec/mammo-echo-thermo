import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from keras import preprocessing
from sklearn import metrics
from scipy import optimize
from .Misc import *

class Join():
	def __init__(self, *models):
		self.models = models
		self.weights = [1/len(self.models) for _ in range(len(self.models))]

	def visualize_heatmap(self, image):
		fig, model_rows = plt.subplots(nrows=len(self.models), ncols=1, constrained_layout=True)
		fig.suptitle(f"Image: {image.image}, Class: {image.category}")

		for model_row in model_rows:
			model_row.remove()

		gridspec = model_rows[0].get_subplotspec().get_gridspec()
		model_rows = [fig.add_subfigure(gs) for gs in gridspec]

		weighted_prediction = np.array([])

		for row, model_row in enumerate(model_rows):
			img = cv.cvtColor(cv.imread(image.image), cv.COLOR_BGR2RGB).astype("float32") * 1./255
			img = np.expand_dims(img, axis=0)
			
			heatmap = self.models[row].compute_heatmap(img)
			jet_heatmap, superimposed_img = self.models[row].get_heatmap(image.image, heatmap)
			predicted = self.models[row].model.predict(img)[0]
			weighted_prediction = np.append(weighted_prediction, predicted[1]*self.weights[row])

			model_row.suptitle(f"Model: {self.models[row].name.title()}")

			ax = model_row.subplots(nrows=1, ncols=4)

			ax[0].imshow(img[0])
			ax[0].set_title("Original")
			ax[0].axis("off")

			ax[1].imshow(self.models[row].filter(img[0][np.newaxis, ...])[0])
			ax[1].set_title("Filter")
			ax[1].axis("off")

			ax[2].imshow(preprocessing.image.array_to_img(jet_heatmap))
			ax[2].set_title(np.round(predicted, 2))
			ax[2].axis("off")
			
			ax[3].imshow(superimposed_img)
			ax[3].set_title(f"Predicted: {np.argmax(predicted)}", color="g" if np.argmax(predicted).astype(str) == image.category else "r")
			ax[3].axis("off")

		plt.show()

		print(f"Weights prediction: {np.round(weighted_prediction, 2)} = {np.round(np.sum(weighted_prediction), 2)}")

	def get_weighted_average(self, generator, iterations=1000, tolerance=1e-7):
		weights = [1/len(self.models) for _ in range(len(self.models))]
		bound_weights = [(0.0, 1.0)  for _ in range(len(self.models))]

		print(f"Weights: {np.round(weights, 2)} -> Accuracy: {np.round(self.get_accuracy(weights, generator), 2)}")

		result = optimize.differential_evolution(self.loss_function, bounds=bound_weights, args=(generator), maxiter=iterations, tol=tolerance)
		weights = self.normalize_weights(result.x)

		print(f"Weights: {np.round(weights, 2)} -> Accuracy: {np.round(self.get_accuracy(weights, generator), 2)}")

		computer.save_plain(f"Weights: {np.round(weights, 2)}")
		
		self.weights = weights

	def get_accuracy(self, weights, generator):
		prediction = np.array([model.model.predict(generator) for model in self.models])
		weighted_prediction = np.tensordot(prediction, weights, axes=((0), (0)))

		return metrics.accuracy_score(generator.labels, np.argmax(weighted_prediction, axis=1))

	def loss_function(self, weights, generator):
		normalize = self.normalize_weights(weights)

		return 1.0 - self.get_weighted_average(normalize, generator)

	def normalize_weights(weights):
		result = np.linalg.norm(weights, 1)

		if result == 0.0:
			return weights

		return weights/result

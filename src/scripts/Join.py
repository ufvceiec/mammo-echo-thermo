import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from keras import preprocessing

class Join():
	def __init__(self, *models):
		self.models = models

	def visualize_heatmap(self, image):
		fig, model_rows = plt.subplots(nrows=len(self.models), ncols=1, constrained_layout=True)
		fig.suptitle(f"Image: {image.image}, Class: {image.category}")

		for model_row in model_rows:
			model_row.remove()

		gridspec = model_rows[0].get_subplotspec().get_gridspec()
		model_rows = [fig.add_subfigure(gs) for gs in gridspec]

		for row, model_row in enumerate(model_rows):
			img = cv.cvtColor(cv.imread(image.image), cv.COLOR_BGR2RGB).astype("float32") * 1./255
			img = np.expand_dims(img, axis=0)
			
			heatmap = self.models[row].compute_heatmap(img)
			jet_heatmap, superimposed_img = self.models[row].get_heatmap(image.image, heatmap)
			predicted = self.models[row].model.predict(img)[0]

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
			ax[3].set_title(f"Predicted: {np.argmax(predicted)}", color="g" if np.argmax(predicted) == int(image.category) else "r")
			ax[3].axis("off")

		plt.show()

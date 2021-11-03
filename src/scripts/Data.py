import os
import pandas as pd

class Data:
	def __init__(self, path):
		self.images = self.__extract_images(path)

	def __extract_images(self, path):
		images = []
		
		for category in os.listdir(path):
			for filename in os.listdir(path + category):
				images.append([path + category + "/" + filename, category])

		return pd.DataFrame(images, columns=["image", "category"])

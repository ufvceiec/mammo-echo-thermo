import tensorflow as tf
import os
import shutil

class computer:
	# This function checks if the model is running on the CPU or GPU
	def check_available_devices(ignore=False):
		print("Available devices:")
		for device in tf.config.experimental.list_physical_devices():
			print(f"- {device.device_type}: {device.name}")

		if (ignore == False) and (len(tf.config.list_physical_devices("GPU")) == 0):
			print("\nCannot run model on GPU. Do you want to run it with the CPU? [Y/N]: ")
			if input().upper() != "Y":
				raise SystemExit("Execution of the model has been canceled!")

	# Function in charge of creating a folder in a path
	def create_folder(path):
		if not os.path.exists(path):
			os.makedirs(path)

	# Function in charge of duplicating a file
	def duplicate_file(source, destination):
		shutil.copy(source, destination)

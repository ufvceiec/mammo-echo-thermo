from copy import copy
import tensorflow as tf
import os
import shutil

class computer:
	# This function checks if the model is running on the CPU or GPU
	def check_available_devices():
		print("Available devices:")
		for device in tf.config.experimental.list_physical_devices():
			print(f"- {device.device_type}: {device.name}")

		if len(tf.config.list_physical_devices("GPU")) == 0:
			print("\nCannot run model on GPU. Do you want to run it with the CPU? [Y/N]: ")
			if input().upper() != "Y":
				raise SystemExit("Execution of the model has been canceled!")

	def create_folder(path):
		if not os.path.exists(path):
			os.makedirs(path)

	def delete_folder(path):
		if os.path.exists(path):
			shutil.rmtree(path)

	def duplicate_file(source, destination):
		shutil.copy(source, destination)

	def save_plain(path, content):
		with open(path, "a") as file:
			file.write(content + "\n")

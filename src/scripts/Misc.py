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

	def create_output_folder(subfolder, new=True):
		if new and os.path.exists("output/" + subfolder):
			shutil.rmtree("output/" + subfolder)

		if not os.path.exists("output"):
			os.makedirs("output")

		if not os.path.exists("output/" + subfolder):
			os.makedirs("output/" + subfolder)

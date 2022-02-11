from scripts import *

computer.check_available_devices(ignore=True)

random_states = [42, 17, 13, 1, 99]

data = Data("./data/")

for random_state in random_states:
	data.training, data.test = data.train_test_split(test_size=0.15, random_state=random_state, stratify=True)

	data.count_labels(data.images, "Original")
	data.count_labels(data.training, "Training")
	data.count_labels(data.test, "Test")

	training_validation_generator = data.training_validation_generator(n_splits=5, random_state=random_state)
	test_generator = data.test_generator()

	filters = {
		"original": lambda x: x,
		"high": lambda x: data.get_image_tensor(x, (330, 0, 0), (360, 255, 255)) + data.get_image_tensor(x, (0, 0, 0), (60, 255, 255)),
		"medium": lambda x: data.get_image_tensor(x, (60, 0, 0), (130, 255, 255)),
		"low": lambda x: data.get_image_tensor(x, (130, 0, 0), (330, 255, 255))
	}

	high_model = [Model(path=f"./output/state_{random_state}/high/fold_{index}", filter=filters["high"]) for index in range(len(training_validation_generator))]
	medium_model = [Model(path=f"./output/state_{random_state}/medium/fold_{index}", filter=filters["medium"]) for index in range(len(training_validation_generator))]
	low_model = [Model(path=f"./output/state_{random_state}/low/fold_{index}", filter=filters["low"]) for index in range(len(training_validation_generator))]

	[high_model[index].compile() for index in range(len(high_model))]
	[medium_model[index].compile() for index in range(len(medium_model))]
	[low_model[index].compile() for index in range(len(low_model))]

	[high_model[index].fit(training_validation_generator[index][0], training_validation_generator[index][1], epochs=600, verbose=False) for index in range(len(high_model))]
	[medium_model[index].fit(training_validation_generator[index][0], training_validation_generator[index][1], epochs=600, verbose=False) for index in range(len(medium_model))]
	[low_model[index].fit(training_validation_generator[index][0], training_validation_generator[index][1], epochs=600, verbose=False) for index in range(len(low_model))]

	for index in range(len(high_model)):
		high_model[index].evaluate(training_validation_generator[index][0], title="train_generator", path=None)
		high_model[index].evaluate(training_validation_generator[index][1], title="validation_generator", path=None)
		high_model[index].evaluate(test_generator, title="test_generator", path=None)

	for index in range(len(medium_model)):
		medium_model[index].evaluate(training_validation_generator[index][0], title="train_generator", path=None)
		medium_model[index].evaluate(training_validation_generator[index][1], title="validation_generator", path=None)
		medium_model[index].evaluate(test_generator, title="test_generator", path=None)

	for index in range(len(low_model)):
		low_model[index].evaluate(training_validation_generator[index][0], title="train_generator", path=None)
		low_model[index].evaluate(training_validation_generator[index][1], title="validation_generator", path=None)
		low_model[index].evaluate(test_generator, title="test_generator", path=None)

	join_high_medium_low = [Join(high_model[index], medium_model[index], low_model[index], path=f"./output/state_{random_state}/join/fold_{index}/high_medium_low") for index in range(len(high_model))]

	[join_high_medium_low[index].get_weighted_average(test_generator, iterations=100, tolerance=1e-7) for index in range(len(high_model))]

	[join_high_medium_low[index].evaluate(test_generator, title="test_generator") for index in range(len(high_model))]

	join_high_medium = [Join(high_model[index], medium_model[index], path=f"./output/state_{random_state}/join/fold_{index}/high_medium") for index in range(len(high_model))]

	[join_high_medium[index].get_weighted_average(test_generator, iterations=100, tolerance=1e-7) for index in range(len(high_model))]

	[join_high_medium[index].evaluate(test_generator, title="test_generator") for index in range(len(high_model))]

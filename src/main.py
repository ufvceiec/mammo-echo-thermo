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
		"original": lambda x: x
	}

	model = [Model(path=f"./output/state_{random_state}/high/fold_{index}", filter=filters["original"]) for index in range(len(training_validation_generator))]

	[model[index].compile() for index in range(len(model))]

	[model[index].fit(training_validation_generator[index][0], training_validation_generator[index][1], epochs=600, verbose=False) for index in range(len(model))]

	for index in range(len(model)):
		model[index].evaluate(training_validation_generator[index][0], title="train_generator", path=None)
		model[index].evaluate(training_validation_generator[index][1], title="validation_generator", path=None)
		model[index].evaluate(test_generator, title="test_generator", path=None)

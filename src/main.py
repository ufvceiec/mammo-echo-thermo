from scripts import *

computer.check_available_devices(ignore=True)

random_state = 42

data = Data("./data/")

data.training, data.test = data.train_test_split(test_size=0.15, random_state=random_state, stratify=True)

data.count_labels(data.images, "Original")
data.count_labels(data.training, "Training")
data.count_labels(data.test, "Test")

training_validation_generator = data.training_validation_generator(n_splits=5, random_state=random_state)
test_generator = data.test_generator()

filters = {
	"original": lambda x: x,
	"red": lambda x: x[:, :, :, 0][..., None],
	"green": lambda x: x[:, :, :, 1][..., None],
	"blue": lambda x: x[:, :, :, 2][..., None]
}

red_model = [Model(path=f"./output/state_{random_state}/red/fold_{index}", filter=filters["red"]) for index in range(len(training_validation_generator))]
green_model = [Model(path=f"./output/state_{random_state}/green/fold_{index}", filter=filters["green"]) for index in range(len(training_validation_generator))]
blue_model = [Model(path=f"./output/state_{random_state}/blue/fold_{index}", filter=filters["blue"]) for index in range(len(training_validation_generator))]

[red_model[index].compile() for index in range(len(red_model))]
[green_model[index].compile() for index in range(len(green_model))]
[blue_model[index].compile() for index in range(len(blue_model))]

[red_model[index].fit(training_validation_generator[index][0], training_validation_generator[index][1], epochs=600, verbose=False) for index in range(len(red_model))]
[green_model[index].fit(training_validation_generator[index][0], training_validation_generator[index][1], epochs=600, verbose=False) for index in range(len(green_model))]
[blue_model[index].fit(training_validation_generator[index][0], training_validation_generator[index][1], epochs=600, verbose=False) for index in range(len(blue_model))]

for index in range(len(red_model)):
	red_model[index].evaluate(training_validation_generator[index][0], title="train_generator", path=None)
	red_model[index].evaluate(training_validation_generator[index][1], title="validation_generator", path=None)
	red_model[index].evaluate(test_generator, title="test_generator", path=None)

for index in range(len(green_model)):
	green_model[index].evaluate(training_validation_generator[index][0], title="train_generator", path=None)
	green_model[index].evaluate(training_validation_generator[index][1], title="validation_generator", path=None)
	green_model[index].evaluate(test_generator, title="test_generator", path=None)

for index in range(len(blue_model)):
	blue_model[index].evaluate(training_validation_generator[index][0], title="train_generator", path=None)
	blue_model[index].evaluate(training_validation_generator[index][1], title="validation_generator", path=None)
	blue_model[index].evaluate(test_generator, title="test_generator", path=None)

join_models = [Join(red_model[index], green_model[index], blue_model[index], path=f"./output/state_{random_state}/join/fold_{index}") for index in range(len(red_model))]

[join_models[index].get_weighted_average(test_generator, iterations=100, tolerance=1e-7) for index in range(len(red_model))]

[join_models[index].evaluate(test_generator, title="test_generator") for index in range(len(red_model))]

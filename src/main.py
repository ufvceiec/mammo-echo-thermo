from scripts import *

data = Data("./data/")

data.training, data.test = data.train_test_split(test_size=0.15, random_state=random_state, shuffle=True, stratify=True)
train_generator, validation_generator, test_generator = data.image_generator(shuffle=False)

filters = {
	"original": lambda x: x,
	"red": lambda x: data.getImageTensor(x, (330, 0, 0), (360, 255, 255)) + data.getImageTensor(x, (0, 0, 0), (60, 255, 255)),
	"green": lambda x: data.getImageTensor(x, (60, 0, 0), (130, 255, 255)),
	"blue": lambda x: data.getImageTensor(x, (130, 0, 0), (330, 255, 255))
}

red_model = Model("red", filter=filters["red"], new=False, summary=False, plot=False)
green_model = Model("green", filter=filters["green"], new=False, summary=False, plot=False)
blue_model = Model("blue", filter=filters["blue"], new=False, summary=False, plot=False)

red_model.compile()
green_model.compile()
blue_model.compile()

red_model.evaluate(train_generator, name="train_generator", path=None)
red_model.evaluate(validation_generator, name="validation_generator", path=None)
red_model.evaluate(test_generator, name="test_generator", path=None)
green_model.evaluate(train_generator, name="train_generator", path=None)
green_model.evaluate(validation_generator, name="validation_generator", path=None)
green_model.evaluate(test_generator, name="test_generator", path=None)
blue_model.evaluate(train_generator, name="train_generator", path=None)
blue_model.evaluate(validation_generator, name="validation_generator", path=None)
blue_model.evaluate(test_generator, name="test_generator", path=None)

join_models = Join(red_model, green_model, blue_model)

# join_models.get_weighted_average(test_generator, iterations=100, tolerance=1e-7)

from scipy.optimize import differential_evolution

def normalize_weights(weights):
	result = np.linalg.norm(weights, 1)

	if result == 0.0:
		return weights

	return weights / result

def loss_function(weights, models, generator):
	normalize = normalize_weights(weights)

	return 1.0 - join_models.get_accuracy(models, normalize, generator)

models = [red_model, green_model, blue_model]

weights = [1/len(models) for _ in range(len(models))]
bound_weights = [(0.0, 1.0)  for _ in range(len(models))]

print("Weights:", weights)
print("Accuracy:", join_models.get_accuracy(models, weights, test_generator))

result = differential_evolution(loss_function, bound_weights, (models, test_generator), maxiter=1000, tol=1e-7)
weights = normalize_weights(result['x'])

print("Weights:", weights)
print("Accuracy:", join_models.get_accuracy(models, weights, test_generator))

computer.save_plain(f"./output/weights.txt", weights)

join_models.evaluate(test_generator, name="test_generator")

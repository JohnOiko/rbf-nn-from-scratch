import time
import numpy as np
from keras.datasets import mnist
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier


# Function that reads the mnist dataset, normalizes it by dividing it by 255, flattens it from 28*28 arrays to 784
# element 1d arrays and returns the data.
def read_normalize_flatten_mnist():
    # Load the dataset using the applicable keras function.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize the train and test data by dividing it by 255.
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Flatten the train and test data from 28*28 2d arrays to 784 element 1d arrays.
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    # Return the data.
    return x_train, y_train, x_test, y_test


# Function that takes the train data and labels and the test data and labels as input, trains the nearest class centroid
# classifier with the train data and labels and prints the train and test data accuracy of the classifier.
def ncc(x_train, y_train, x_test, y_test):
    # Create the nearest class centroid classifier and train it with the train data and labels.
    ncc_classifier = NearestCentroid()
    print(f"NCC calculations started...")
    ncc_classifier.fit(x_train, y_train)
    # Calculate and print the train and test data accuracies.
    training_accuracy = ncc_classifier.score(x_train, y_train)
    print(f"-Train accuracy: {training_accuracy} ({round(training_accuracy * 100, 2)}%).")
    test_accuracy = ncc_classifier.score(x_test, y_test)
    print(f"-Test accuracy: {test_accuracy} ({round(test_accuracy * 100, 2)}%).\n")


# Function that takes the train data and labels, the test data and labels and the number of neighbors k as input, trains
# the k nearest neighbors classifier with the train data and labels for the k value given as input and prints the train
# and test data accuracy of the classifier.
def knn(x_train, y_train, x_test, y_test, k):
    # Create the k nearest neighbor classifier and train it with the train data and labels.
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    print(f"KNN calculations for k={k} started...")
    knn_classifier.fit(x_train, y_train)
    # Calculate and print the train and test data accuracies.
    training_accuracy = knn_classifier.score(x_train, y_train)
    print(f"-Train accuracy: {training_accuracy} ({round(training_accuracy * 100, 2)}%).")
    test_accuracy = knn_classifier.score(x_test, y_test)
    print(f"-Test accuracy: {test_accuracy} ({round(test_accuracy * 100, 2)}%).\n")


# Trains and returns a custom RBF neural network using the RBFNetwork class with a 784 neuron input layer, a 1568 neuron
# hidden RBF layer and a 10 neuron output layer, one output neuron per class/digit.
def custom_rbf_neural_network(x_train, y_train, x_test, y_test):
    # Create the RBF neural network using the RBFNetwork class.
    custom_rbf_network = RBFNetwork(learning_rate=9.0, momentum=0.9, decay=0.0001)

    # Add an input layer which takes 784 inputs and has 784 neurons.
    custom_rbf_network.add_layer(784, 784, "Input")
    # Add a hidden RBF layer which takes 784 inputs, has 1568 (784 * 2) neurons and uses the random center
    # initialization training method.
    custom_rbf_network.add_layer(784, 1568, "RBF", init="random")
    # Add an output layer which takes 1568 inputs and has 10 neurons/outputs (one for each class/digit).
    custom_rbf_network.add_layer(1568, 10, "Output")

    print(f"RBF neural network calculations started...\n")
    # Train the model for 50 epochs with a batch size of 128 and save the training time.
    training_time = custom_rbf_network.fit(x_train, y_train, epochs=50, epoch_print_rate=1, batch_size=128)

    # Print the training time.
    print(f"\nTime elapsed for training: {round(training_time, 2)} seconds.")
    # Test the model with the train data and print the results.
    print("Train data results:")
    custom_rbf_network.evaluate(x_train, y_train, print_results=True)
    # Test the model with the test data and print the results.
    print("Test data results:")
    custom_rbf_network.evaluate(x_test, y_test, print_results=True)

    # Returns the RBF neural network so that it can be used outside this function.
    return custom_rbf_network


# Function that runs the code of the third project. Specifically, the first four parameters are used as train and test
# data and labels. Additionally, if any of the parameters ncc_test, knn_test or custom_rbf_nn_test are false, then the
# respective model's performance is not measured and printed. By default, all three models are tested.
def third_project(x_train, y_train, x_test, y_test, ncc_test=True, knn_test=True, custom_rbf_nn_test=True):
    # If enabled, measure and print the performance of the nearest class centroid classifier.
    if ncc_test:
        ncc(x_train, y_train, x_test, y_test)

    # If enabled, measure and print the performance of the k nearest neighbor classifier for k=1 and k=3.
    if knn_test:
        for k in [1, 3]:
            knn(x_train, y_train, x_test, y_test, k)

    # If enabled, measure and print the performance and training time of the custom RBF neural network.
    if custom_rbf_nn_test:
        custom_rbf_neural_network(x_train, y_train, x_test, y_test)


# Function that calculates and returns the average accuracy given the outputs of a model and the true values.
def calculate_accuracy(output, y_true):
    predictions = np.argmax(output, axis=1)
    return np.mean(predictions == y_true)


# Function that finds a wrong prediction of the given model for the input data x and labels y and returns its index.
def find_wrong_prediction(rbf_nn, x, y, print_all_predictions=False):
    print("\nLooking for a wrong prediction in the neural network:")
    # The current sample (starting from zero).
    sample = 0
    # The probabilities output of the model for the current sample.
    probabilities = rbf_nn.predict(x[sample])
    # The most likely prediction based on the model's output probabilities.
    prediction = np.argmax(probabilities)
    # The label of the current sample.
    label = y[sample]

    # Check every sample until a wrong prediction is found.
    while sample < len(y) and prediction == label:
        if print_all_predictions:
            print(f"\nSample {sample + 1}: prediction is {prediction}, label is {label}.")
        # Move on to the next sample.
        sample += 1
        probabilities = rbf_nn.predict(x[sample])
        prediction = np.argmax(probabilities)
        label = y[sample]

    # If a wrong prediction was found, print the results and its probabilities outputted by the model.
    if prediction != label:
        print(f"\nSample {sample + 1}: prediction is {prediction}, label is {label}.")
        print(f"Here is the model's output for sample {sample + 1}:")
        print(probabilities)
        return sample
    else:
        print("No wrong predictions were found.")
        return -1


# Parent class that represents a general layer of neurons.
class Layer:
    # Constructor that initializes the layer's output to None.
    def __init__(self):
        # Saves the layer's output.
        self.output = None

    # Method that executes a forward pass with the given inputs.
    def forward_pass(self, inputs):
        pass


# Class that implements an input layer and inherits the Layer class.
class RBFInputLayer(Layer):
    def forward_pass(self, inputs):
        # Calculate the input layer's output which is its input.
        self.output = inputs


# Class that implements a hidden RBF layer and inherits the Layer class.
class RBFHiddenLayer(Layer):
    # Constructor that initializes the layer's members.
    def __init__(self, neuron_num, init="random"):
        super().__init__()
        # Initialize the hidden RBF layer's centers to None.
        self.centers = None
        # Initialize the hidden RBF layer's sigmas to None.
        self.sigmas = None
        # Save the hidden RBF layer's number of neurons.
        self.neuron_num = neuron_num
        # Save the hidden RBF layer's type of center initialization, defaults to "random", other option is "k-means".
        self.init = init

    # Method that trains the hidden RBF layer by calculating its centers and sigmas based on the given inputs.
    def train(self, inputs):
        # If the hidden RBF layer's type of center initialization is "random", sample neuron_num number of input samples
        # and save them as centers.
        if self.init == "random":
            random_idx = np.array([*range(len(inputs))])
            np.random.shuffle(random_idx)
            self.centers = inputs[random_idx[range(self.neuron_num)]]

        # Else if the hidden RBF layer's type of center initialization is not "random", use the k-means algorithm
        # provided by the sklearn library to cluster the input samples to neuron_num number of clusters and save the
        # clusters' centers as the hidden RBF layer's centers. The k-means algorithm's max iterations are capped to
        # prevent it from taking too long to complete the clustering during the hidden RBF layer's training.
        else:
            k_means_cluster = KMeans(n_clusters=self.neuron_num, init="random", max_iter=5)
            k_means_cluster.fit(inputs)
            self.centers = k_means_cluster.cluster_centers_

        # Calculate the sigma value which is equal among all neurons as the maximum distance between centers divided by
        # the quare root of double the number of centers as shown on slide 16 of mister Diamantaras' slides on RBFs in
        # the e-Learning page of the class. The maximum distance between centers is calculated with the cdist function
        # of the library scipy using the Manhattan/city block distance to lower the calculation time.
        sigma = np.max(cdist(self.centers, self.centers, metric="cityblock")) / np.sqrt(2 * self.neuron_num)
        # For each neuron in the hidden RBF layer, set its sigma value to the one calculated in the previous line.
        self.sigmas = np.full(self.neuron_num, sigma)

    def forward_pass(self, inputs):
        # Calculate and save the hidden RBF layer's output using the Gauss RBF function as shown on slide 3 of mister
        # Diamantaras' slides on RBFs in the e-Learning page of the class. Again the cdist function of the library scipy
        # is used to calculate the euclidean distance/2-norm to lower the calculation time.
        self.output = np.exp(-np.power(cdist(inputs, self.centers, metric="euclidean"), 2) / np.power(self.sigmas, 2))


# Class that implements an output layer and inherits the Layer class.
class RBFOutputLayer(Layer):
    # Constructor that initializes the output layer's members.
    def __init__(self, input_num, neuron_num):
        super().__init__()
        # Biases initialized to 0.
        self.biases = np.zeros((1, neuron_num))
        # Weights initialized to random values using the standard normal distribution provided by numpy and multiplied
        # by a small factor to keep the initial weights small.
        self.weights = 0.01 * np.random.randn(input_num, neuron_num)
        # Saves the inputs of the layer.
        self.inputs = None
        # Save the gradients with respect to the corresponding parameter.
        self.inputs_gradients = None
        self.biases_gradients = None
        self.weights_gradients = None
        # Initialize the biases momenta for the optimizer to zeroes.
        self.biases_momenta = np.zeros_like(self.biases)
        # Initialize the weights momenta for the optimizer to zeroes.
        self.weights_momenta = np.zeros_like(self.weights)

    def forward_pass(self, inputs):
        # Save the inputs for back propagation.
        self.inputs = inputs
        # Calculate the output values based on the inputs, the weights and the biases.
        self.output = np.dot(inputs, self.weights) + self.biases

    def back_propagate(self, next_layer_gradients):
        # Calculate the gradients with respect to the layer's biases.
        self.biases_gradients = np.sum(next_layer_gradients, axis=0, keepdims=True)
        # Calculate the gradients with respect to the layer's weights. The inputs' matrix must be transposed to make the
        # dimensions of the dot product fit.
        self.weights_gradients = np.dot(self.inputs.T, next_layer_gradients)


# Parent class that represents an activation function.
class ActivationFunction:
    def __init__(self):
        # Saves the inputs of the activation function.
        self.inputs = None
        # Saves the output of the activation function.
        self.output = None
        # Saves the gradients with respect to the activation function's inputs.
        self.inputs_gradients = None

    # Method that executes a forward pass with the given inputs.
    def forward_pass(self, inputs):
        pass

    # Method that executes back propagation given the gradients returned by the loss function with respect to its
    # inputs.
    def back_propagate(self, next_layer_gradients):
        pass


# Class that implements the SoftMax activation function and inherits the ActivationFunction class.
class SoftMax(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward_pass(self, inputs):
        # Save the inputs for back propagation.
        self.inputs = inputs
        # Calculate the output using the inputs based on the SoftMax formula.
        # First calculate the exponential inputs.
        exponential_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Then calculate the sum of the exponential inputs.
        exponential_inputs_sum = np.sum(exponential_inputs, axis=1, keepdims=True)
        # Lastly, normalize the exponential inputs.
        self.output = exponential_inputs/exponential_inputs_sum

    def back_propagate(self, next_layer_gradients):
        # Make an empty array with the dimensions of the next layer's gradients.
        self.inputs_gradients = np.empty_like(next_layer_gradients)

        # For loop which iterates each of the output and gradients.
        for index, (single_output, single_next_layer_gradient) in enumerate(zip(self.output, next_layer_gradients)):
            # Reshape the single input using reshape(-1, 1) to have only one column and as many rows as needed.
            reshaped_single_output = single_output.reshape(-1, 1)
            # Create a diagonal array whose diagonal values are the single output's values.
            diagonal_single_output = np.diagflat(reshaped_single_output)
            # Calculate the Jacobian matrix of the single output using the previous diagonal array and the reshaped
            # single output.
            jacobian_matrix = diagonal_single_output - np.dot(reshaped_single_output, reshaped_single_output.T)
            # Calculate the gradient and add it to the array of gradients.
            self.inputs_gradients[index] = np.dot(jacobian_matrix, single_next_layer_gradient)


# Class that implements the stochastic gradient descent optimizer.
class SgdOptimizer:
    # Constructor that initializes the optimizer's parameters. The default values of the parameters are the same as
    # keras' default values.
    def __init__(self, learning_rate=0.01, momentum=0.0, decay=0.0):
        # Learning rate, defaults to 0.01.
        self.learning_rate = learning_rate
        # Momentum, defaults to 0.0.
        self.momentum = momentum
        # Decay rate, defaults to 0.0.
        self.decay = decay
        # Current learning rate, is initialized to the given learning rate.
        self.current_learning_rate = learning_rate
        # Iteration counter.
        self.iteration_counter = 0

    # Updates the learning rate if the decay is enabled (it is different from 0.0). Must be called once before any
    # parameter update.
    def update_learning_rate(self):
        # If the decay is enabled, calculate the current learning rate using the decay formula.
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration_counter))

    # Updates the weights and biases of the given layer.
    def update_layer_parameters(self, layer):
        # Calculate the needed changes in biases and weights. If the momentum is not 0.0 (it is enabled), it is used in
        # the calculations, else if it is 0.0 (it is disabled), that part of the calculation becomes 0 and thus doesn't
        # affect it.
        bias_updates = self.momentum * layer.biases_momenta - self.current_learning_rate * layer.biases_gradients
        weight_updates = self.momentum * layer.weights_momenta - self.current_learning_rate * layer.weights_gradients

        # If the momentum is enabled (it is not 0.0), update the momenta for the biases and the weights.
        if self.momentum:
            layer.biases_momenta = bias_updates
            layer.weights_momenta = weight_updates

        # Update the layer's biases and weights using the previously calculated bias updates and weight updates.
        layer.biases += bias_updates
        layer.weights += weight_updates

    # Increments the iteration counter. Must be called once after any parameter update.
    def increment_iteration_counter(self):
        self.iteration_counter += 1


# Parent class that represents a loss function.
class Loss:
    # Constructor for the class.
    def __init__(self):
        # The gradients of the loss function with respect to the loss function's inputs.
        self.inputs_gradients = None

    # Calculates the average loss of all the given samples, given a model's output and the true values.
    def calculate_loss(self, output, y):
        # Calculate the loss for each sample.
        individual_losses = self.forward_pass(output, y)
        # Calculate and return the average loss.
        return np.mean(individual_losses)


# Class that represents the categorical cross entropy loss function and inherits the Loss function.
class CategoricalCrossEntropy(Loss):
    # Method that executes a forward pass given a model's predictions and the true values and returns the loss for each
    # sample.
    def forward_pass(self, y_pred, y_true):
        # Calculate the number of samples in the given predictions.
        sample_num = len(y_pred)
        # Clip the predictions to prevent division by 0.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Save the probabilities for the true values in a 1d array, using different methods based on if the true values
        # are saved as categorical labels (1d array) or one-hot encoded labels (2d array).
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(sample_num), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Calculate and return the loss for each sample using the negative logarithm of the correct confidences.
        return -np.log(correct_confidences)

    # Method that executes back propagation given the gradients and the true values.
    def back_propagate(self, next_layer_gradients, y_true):
        # Calculate the number of samples in the given gradients.
        sample_num = len(next_layer_gradients)
        # Calculate the number of labels in each sample using the first sample.
        label_num = len(next_layer_gradients[0])

        # If the true labels are sparse, turn them into a one-hot vector.
        if len(y_true.shape) == 1:
            y_true = np.eye(label_num)[y_true]

        # Calculate the gradient with respect to the inputs.
        self.inputs_gradients = - y_true / next_layer_gradients
        # Normalize the previously calculated gradient.
        self.inputs_gradients = self.inputs_gradients / sample_num


# Class that merges the SoftMax activation function and the categorical cross entropy loss function to avoid division by
# zero.
class SoftmaxCategoricalCrossEntropy:
    # Initializer that creates the SoftMax activation function and categorical cross entropy loss function objects.
    def __init__(self):
        self.activation_function = SoftMax()
        self.loss_function = CategoricalCrossEntropy()
        self.output = None
        self.inputs_gradients = None

    # Method that executes a forward pass given the inputs and the true values and returns the average loss value.
    def forward_pass(self, inputs, y_true):
        # Calculate the output layer's activation function output.
        self.activation_function.forward_pass(inputs)
        # Save the previously calculated output.
        self.output = self.activation_function.output
        # Calculate and return the average loss value while executing a forward pass through the output layer's loss
        # function.
        return self.loss_function.calculate_loss(self.output, y_true)

    # Method that executes back propagation given the gradients returned by the next layer and the true values.
    def back_propagate(self, next_layer_gradients, y_true):
        # Calculate the number of samples in the given gradients.
        samples_num = len(next_layer_gradients)

        # If labels are one-hot encoded, turn them into categorical labels.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Make a copy of the next layer's gradients so that it can be edited.
        self.inputs_gradients = next_layer_gradients.copy()
        # Calculate and save the gradient.
        self.inputs_gradients[range(samples_num), y_true] -= 1
        # Normalize the previously calculated gradient.
        self.inputs_gradients = self.inputs_gradients / samples_num


# Class that represents a whole RBF neural network.
class RBFNetwork:
    # Initializer that takes as input the optimizer type and it's parameters and initializes the optimizer, the list
    # of layers and activation/loss function of the model.
    def __init__(self, optimizer="SGD", learning_rate=9.0, momentum=0.9, decay=0.0001):
        # The list of layers of the model.
        self.layers = []
        # The activation and loss functions of the model's output layer bundled together in one class.
        self.output_activation_loss_function = SoftmaxCategoricalCrossEntropy()
        # If the chosen optimizer is the stochastic gradient descent, create it with the given parameters.
        if optimizer == "SGD":
            self.optimizer = SgdOptimizer(learning_rate, momentum, decay)

    # Method that adds a new layer to the model taking its number of inputs, number of neurons, layer type and center
    # initialization method as input. The center initialization method defaults to "random" and only has effect if the
    # selected layer type is "RBF".
    def add_layer(self, input_num, neuron_num, layer_type, init="random"):
        # Create and add the corresponding layer to the model's layers list based on its given type.
        if layer_type == "Input":
            self.layers.append(RBFInputLayer())
        elif layer_type == "RBF":
            self.layers.append(RBFHiddenLayer(neuron_num, init=init))
        else:
            self.layers.append(RBFOutputLayer(input_num, neuron_num))

    # Method that trains the model with the given input data (x) and labels (y). The parameters are the number of
    # epochs, the epoch print rate which changes how often progress is printed and the batch size which defaults to 128.
    # These parameters are used for training the model's output layer using Stochastic Gradient Descent.
    def fit(self, x, y, epochs=1, epoch_print_rate=1, batch_size=128):
        # The fist dimension of the input data (the number of training samples).
        sample_num = len(x)
        # The amount of batches that will be created (or the amount of batches -1).
        batch_num = sample_num // batch_size
        # The total training time without counting the evaluation time when printing progress.
        training_time = 0
        # Save the starting time of the hidden RBF layer's training.
        hidden_layer_start_time = time.time()

        # Save the model's input layer.
        input_layer = self.layers[0]
        # Save the model's hidden RBF layer.
        hidden_layer = self.layers[1]

        # Do a forward pass of the training samples through the input layer.
        input_layer.forward_pass(x)
        # Train the hidden RBF layer (calculate its centers and sigmas) using the input layer's output.
        hidden_layer.train(input_layer.output)
        # Do a forward pass of the input layer's output through the hidden RBF layer.
        hidden_layer.forward_pass(input_layer.output)

        # Add the hidden RBF layer's training time to the total training time.
        training_time += time.time() - hidden_layer_start_time
        # The training time for the last epoch_print_rate number of epochs without counting the evaluation time when
        # printing progress.
        print_rate_epochs_training_time = 0

        # For loop which loops for the given amount of epochs.
        for epoch in range(epochs):
            # Save the start time of the epoch.
            epoch_start_time = time.time()

            # For loop which loops for all the batches.
            for batch_index in range(batch_num):
                # If this is the last batch, save the corresponding input and label values from the start of this batch
                # to the last input and label. The output of the hidden RBF layer is used as input.
                if batch_index == batch_num - 1:
                    current_input = hidden_layer.output[range(batch_index * batch_size, batch_num * batch_size), :]
                    current_y = y[range(batch_index * batch_size, batch_num * batch_size)]

                # Else if this isn't the last batch, save the corresponding input and label values from the start of
                # this batch to the start of the next batch minus one. The output of the hidden RBF layer is used as
                # input.
                else:
                    current_input = hidden_layer.output[range(batch_index * batch_size,
                                                              (batch_index + 1) * batch_size), :]
                    current_y = y[range(batch_index * batch_size, (batch_index + 1) * batch_size)]

                # Save the model's output layer.
                output_layer = self.layers[2]
                # Do a forward pass of the current batch of the hidden RBF layer's output through the output layer.
                output_layer.forward_pass(current_input)
                # Do a forward pass of the current batch of the output layer's output through its activation and loss
                # functions which are bundled together.
                self.output_activation_loss_function.forward_pass(output_layer.output, current_y)

                # Do back propagation through the output layer's activation and loss functions using their output and
                # the current batch of labels.
                self.output_activation_loss_function.back_propagate(self.output_activation_loss_function.output,
                                                                    current_y)
                # Do back propagation through the output layer using the gradients with respect to inputs of the
                # current activation and loss functions which were just calculated by the previous back propagation.
                output_layer.back_propagate(self.output_activation_loss_function.inputs_gradients)

                # This is where the optimization is done.
                # Update the learning rate of the optimizer before optimizing.
                self.optimizer.update_learning_rate()
                # Optimize the output layer using the Stochastic Gradient Descent optimizer.
                self.optimizer.update_layer_parameters(output_layer)
                # Increment the iteration counter of the optimizer.
                self.optimizer.increment_iteration_counter()

            # Save the finish time of the epoch and add the epoch's training time to the total training time and the
            # training time for the last epoch_print_rate number of epochs.
            epoch_finish_time = time.time()
            training_time += epoch_finish_time - epoch_start_time
            print_rate_epochs_training_time += epoch_finish_time - epoch_start_time

            # If it's time to print the progress given the epoch print rate, or it's the last epoch, print the progress.
            if (epoch + 1) % epoch_print_rate == 0 or epoch == epochs - 1:
                # Evaluate the whole data to get the average loss and accuracy.
                loss, accuracy = self.evaluate(x, y, print_results=False, recalculate_hidden_layer=False)

                # If this is the last epoch and the number of epochs is not perfectly dividable by the epoch_print_rate,
                # save the average epoch training time as the training time for the last epochs divided by the number of
                # last remaining epochs. For example, if there are 50 epochs and the epoch_print_rate is 8, the divisor
                # will be the remainder of 50/8 which is 2.
                if epoch == epochs - 1 and epochs % epoch_print_rate != 0:
                    print_rate_epochs_average_training_time = print_rate_epochs_training_time / (epochs
                                                                                                 % epoch_print_rate)
                # Else save the average epoch training time as the training time for the last epoch_print_rate number of
                # epochs divided by the epoch_print_rate.
                else:
                    print_rate_epochs_average_training_time = print_rate_epochs_training_time / epoch_print_rate
                # Reset the training time for the last epoch_print_rate number of epochs.
                print_rate_epochs_training_time = 0

                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"loss: {loss:.4f} - accuracy: {accuracy:.4f} - " +
                      f"{print_rate_epochs_average_training_time * 1000:.0f}ms/epoch")

        return training_time

    # Method that evaluates the given input data (x) and labels (y) and returns the average loss and accuracy.
    def evaluate(self, x, y, print_results=False, recalculate_hidden_layer=True):
        # Save the model's hidden RBF layer.
        hidden_layer = self.layers[1]
        # Save the model's output layer.
        output_layer = self.layers[2]

        # If the option to do a forward pass through the hidden RBF layer with the new input samples x is selected
        # (recalculate_hidden_layer), then do so.
        if recalculate_hidden_layer:
            # Save the model's input layer.
            input_layer = self.layers[0]
            # Do a forward pass of the input samples through the input layer.
            input_layer.forward_pass(x)
            # Do a forward pass of the input layer's output through the hidden RBF layer.
            hidden_layer.forward_pass(input_layer.output)

        # Do a forward pass of the hidden RBF layer's output through the output layer.
        output_layer.forward_pass(hidden_layer.output)
        # Do a forward pass of the output layer's output through its activation and loss functions which are bundled
        # together and save the loss.
        loss = self.output_activation_loss_function.forward_pass(output_layer.output, y)

        # Calculate and save the average accuracy of the model using the applicable function, the output layer's
        # activation and loss functions' output and the labels y.
        accuracy = calculate_accuracy(self.output_activation_loss_function.output, y)

        # If enabled, print the results.
        if print_results:
            print(f"loss: {loss:.4f} - accuracy: {accuracy:.4f}")

        # Return the calculated average loss and accuracy.
        return loss, accuracy

    # Method that returns the model's prediction for the given input x which can be a single sample in a one-dimensional
    # array or multiple samples in a two-dimensional array.
    def predict(self, x):
        # Save the model's input layer.
        input_layer = self.layers[0]
        # Save the model's hidden RBF layer.
        hidden_layer = self.layers[1]
        # Save the model's output layer.
        output_layer = self.layers[2]

        # Do a forward pass of the input through the input layer. If the input is a single sample, pass it as a
        # two-dimensional array, else if it is multiple samples, pass it as is because it is already a two-dimensional
        # array.
        input_layer.forward_pass([x] if x.ndim == 1 else x)
        # Do a forward pass of the input layer's output through the hidden RBF layer.
        hidden_layer.forward_pass(input_layer.output)
        # Do a forward pass of the hidden RBF layer's output through the output layer.
        output_layer.forward_pass(hidden_layer.output)
        # Do a forward pass of the output layer's output through its activation and loss functions which are bundled
        # together with zero as the label (the label is not needed in the calculations of the output).
        self.output_activation_loss_function.forward_pass(output_layer.output, np.array([0]))
        # Return the output of the output layer's activation and loss functions which are bundled together.
        return self.output_activation_loss_function.output


# Read the normalized flattened mnist dataset.
X_train, Y_train, X_test, Y_test = read_normalize_flatten_mnist()

# Run the third project's code with the two classifiers and the custom RBF neural network as described in the function's
# comments.
third_project(X_train, Y_train, X_test, Y_test, ncc_test=True, knn_test=True, custom_rbf_nn_test=True)

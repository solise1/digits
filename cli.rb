require_relative "network"
require_relative "mnist_loader"

def run(epochs, mini_batch_size, eta, lmbda = 0.0, load_test_data: true, **opts)
  training_data = MnistLoader.load_training_data

  puts "Training data loaded!\n\n"

  test_data = MnistLoader.load_test_data if load_test_data

  puts "Test data loaded!\n\n" if load_test_data

  net = Network.new([784, 30, 10])

  puts "Starting Stochastic Gradient Descent for Network #{net.layers}\n\n"

  net.sgd(training_data, epochs, mini_batch_size, eta, lmbda, test_data, **opts)
end

opts = {
#  monitor_training_cost: true,
  monitor_training_accuracy: true,
#  monitor_evaluation_cost: true,
  monitor_evaluation_accuracy: true,
  save: true
}

run(100, 10, 0.5, 0.1, **opts)


#run(2, 10, 3.0)

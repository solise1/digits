require_relative "network"
require_relative "mnist_loader"

def run(epochs, mini_batch_size, eta, load_test_data: true)
  training_data = MnistLoader.load_training_data

  test_data = MnistLoader.load_test_data if load_test_data

  net = Network.new([784, 30, 10])

  net.sgd(training_data, epochs, mini_batch_size, eta, test_data)
end

run(10, 10, 3.0)


#run(2, 10, 3.0)

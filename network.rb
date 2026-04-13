require "json"

require "numo/narray/alt"
require "numo/linalg"

class Network
  attr_reader :num_layers, :layers, :biases, :weights

  def self.load(filename)
    f = File.open(filename, "r")
    data = JSON.load f
    f.close

    new(
      data["layers"],
      biases: Numo::DFloat[data["biases"]],
      weights: Numo::DFloat[data["weights"]]
    )
  end

  def initialize(layers, biases: nil, weights: nil)
    @layers = layers
    @num_layers = layers.size
    @biases = biases || layers[1..].map { |y| Numo::DFloat.new(y, 1).rand_norm }
    @weights = weights || layers[..-2].zip(layers[1..]).map { |x, y| Numo::DFloat.new(y, x).rand_norm / Numo::NMath.sqrt(x) }
  end

  # ? Verify that this updates correctly every step
  def feedforward(a)
    @biases.zip(@weights).each do |b, w|
      a = sigmoid(Numo::Linalg.dot(w, a) + b)
    end
    a
  end

  # * Stochastic Gradient Descent
  # * training_data is a matrix of which includes the inputs and the
  # * desired outputs, eta is the learning rate, test_data is optional
  # * as it slows things down but it can allow you to track progress
  def sgd(training_data, epochs, mini_batch_size, eta, lmbda = 0.0, test_data=nil, **opts)
    training_cost = []
    training_accuracy = []
    evaluation_cost = []
    evaluation_accuracy = []
    training_data_size = training_data.size
    test_data_size = test_data&.size

    epochs.times do |j|
      training_data.shuffle!

      training_data.each_slice(mini_batch_size) { |mb| update_weight_and_bias(mb, eta, lmbda, training_data_size) }

      puts "\nEpoch #{j} - training complete"

      if opts[:monitor_training_cost]
        training_cost << total_cost(training_data, lmbda)
        puts "Cost on training data: #{training_cost[-1]}"
      end

      if opts[:monitor_training_accuracy]
        training_accuracy << accuracy(training_data)
        puts "Accuracy on training data: #{(training_accuracy[-1].fdiv(training_data_size) * 100).round(2)}%"
      end

      if opts[:monitor_evaluation_cost]
        evaluation_cost << total_cost(test_data, lmbda)
        puts "Cost on test data: #{evaluation_cost[-1]}"
      end

      if opts[:monitor_evaluation_accuracy]
        evaluation_accuracy << accuracy(test_data)
        puts "Accuracy on test data: #{(evaluation_accuracy[-1].fdiv(test_data_size) * 100).round(2)}%"
      end
    end

    if opts[:save]
      save("net.json")
    end
  end

  # * eta is the learning rate
  # * lmbda is the regularization parameter
  # * n is the training_data_size
  def update_weight_and_bias(mini_batch, eta, lmbda, n)
    nabla_b = biases.map { |b| Numo::DFloat.zeros(b.shape) }
    nabla_w = weights.map { |w| Numo::DFloat.zeros(w.shape) }

    mini_batch.each do |x, y|
      delta_nabla_b, delta_nabla_w = backpropagate(x, y)
      nabla_b = nabla_b.zip(delta_nabla_b).map { |nb, dnb| nb + dnb }
      nabla_w = nabla_w.zip(delta_nabla_w).map { |nw, dnw| nw + dnw }
    end

    @weights = weights.zip(nabla_w).map do |w, nw|
      (1 - eta * (lmbda / n)) * w - (eta / mini_batch.size) * nw
    end

    @biases = biases.zip(nabla_b).map do |b, nb|
      b - (eta / mini_batch.size) * nb
    end
  end

  def backpropagate(x, y)
    nabla_b = biases.map { |b| Numo::DFloat.zeros(b.shape) }
    nabla_w = weights.map { |w| Numo::DFloat.zeros(w.shape) }

    activations, z_vectors = calculate_activations_and_z_vectors(x)

    delta = activations[-1] - y
    nabla_b[-1] = delta
    nabla_w[-1] = Numo::Linalg.dot(delta, activations[-2].transpose)

    (2...num_layers).each do |l|
      delta = Numo::Linalg.dot(weights[-l + 1].transpose, delta) * sigmoid_derivative(z_vectors[-l])
      nabla_b[-l] = delta
      nabla_w[-l] = Numo::Linalg.dot(delta, activations[-l-1].transpose)
    end

    [nabla_b, nabla_w]
  end

  def accuracy(data)
    data.count { |x, y| feedforward(x).max_index == y.max_index }
  end

  def total_cost(data, lmbda)
    data.sum { |x, y| cost(feedforward(x), y) / data.size } +
      0.5 * (lmbda / data.size) * weights.map { |w| Numo::Linalg.norm(w) ** 2 }.sum
  end

  def save(filename)
    data = {
      layers: layers,
      weights: weights.map(&:to_a),
      biases: biases.map(&:to_a)
    }

    File.open(filename, "w") do |f|
      f.write(JSON.dump(data))
    end
  end

  private
    def sigmoid(z)
      1.0 / (1.0 + Numo::NMath.exp(-z))
    end

    def sigmoid_derivative(z)
      sz = sigmoid(z)
      sz * (1 - sz)
    end

    # * Cross-Entropy Cost function
    def cost(a, y)
      (-y * Numo::NMath.log(a) - (1 - y) * Numo::NMath.log(1 - a)) #.sum
    end

    def calculate_activations_and_z_vectors(x)
      activation = x
      activations = [x]
      z_vectors = []

      biases.zip(weights).each do |b, w|
        z = Numo::Linalg.dot(w, activation) + b

        z_vectors << z

        activation = sigmoid(z)
        activations << activation
      end

      [activations, z_vectors]
    end
end

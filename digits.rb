require "numo/random" # includes numo/narray/alt so no need to import twice

class Network
  attr_reader :num_layers, :sizes, :biases, :weights

  def initialize(sizes)
    @num_layers = sizes.size
    @sizes = sizes
    @biases = sizes[1..].map { |y| rng.normal(shape: [y, 1]) }
    @weights = sizes[..-2].zip(sizes[1..]).map { |x, y| rng.normal(shape: [y, x]) }
  end

  def rng
    @rng ||= Numo::Random::Generator.new
  end

  def sigmoid(z)
    1.0 / (1.0 + Numo::Math.exp(-z))
  end

  def sigmoid_derivative(z)
    sz = sigmoid(z)
    sz * (1 - sz)
  end

  def feedforward(a)
    biases.zip(weights).each do |b, w|
      a = sigmoid(w.dot(a) + b)
    end
    a
  end

  # Stochastic Gradient Descent
  # training_data is a matrix of which includes the inputs and the
  # desired outputs, eta is the learning rate, test_data is optional
  # as it slows things down but it can allow you to track progress
  def sgd(training_data, epochs, mini_batch_size, eta, test_data=nil)
    epochs.times do |j|
      training_data.shuffle

      mini_batches = training_data.each_slice(mini_batch_size).to_a

      mini_batches.each { |mb| update_weight_and_bias(mb, eta / mini_batch_size) }

      if test_data
        puts "Epoch #{j}: #{evaluate(test_data)} / #{training_data.size}"
      else
        puts "Epoch #{j} complete!"
      end
    end
  end

  def update_weight_and_bias(mini_batch, learning_rate)
    nabla_b = biases.map { |b| Numo::DFloat.zeros(b.shape) }
    nabla_w = weights.map { |w| Numo::DFloat.zeros(w.shape) }

    mini_batch.each do |x, y|
      delta_nabla_b, delta_nabla_w = backprop(x, y)
      nabla_b = nabla_b.zip(delta_nabla_b).map(&:sum)
      nabla_w = nabla_w.zip(delta_nabla_w).map(&:sum)
    end

    @weights = weights.zip(nabla_w).map { |w, nw| w - learning_rate * nw }
    @biases = biases.zip(nabla_b).map { |b, nb| b - learning_rate * nb }
  end

  # Computes the gradient for the cost function
  def backprop(x, y)
    nabla_b = biases.map { |b| Numo::DFloat.zeros(b.shape) }
    nabla_w = weights.map { |w| Numo::DFloat.zeros(w.shape) }

    activation = x
    activations = [x]
    z_vectors = []

    biases.zip(weights).each do |b, w|
      z = w.dot(x) + b

      z_vectors << z

      activation = sigmoid(z)
      activations << activation
    end

    delta = cost_derivative(activations[-1], y) * sigmoid_derivative(z_vector[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = delta.dot(activations[-2].transpose)

    (2...num_layers).each do |l|
      z = zs[-l]
      sp = sigmoid_derivative(z)
      delta = weights[-l + 1].transpose.dot(delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = delta.dot(activations[-l-1].transpose
    end

    [nabla_b, nabla_w]
  end
end

require_relative "network"
require_relative "mnist_loader"

# * Neural network CLI
# * You can pass any values listed in VALID_INPUTS,
# * or run as is to get default behavior. Values
# * not listed in that constant are ignored.
class CLI
  VALID_INPUTS = (%i[load_network layers epochs mini_batch_size eta lmbda skip_tests] + Network::VALID_SGD_OPTS).to_set

  attr_reader :opts, :training_data, :test_data, :epochs, :mini_batch_size, :eta, :lmbda

  def self.run
    options = ARGV.map { |a| a.split("=") }
                  .to_h { |a, b| [a, b || true] }
                  .transform_keys { |k| k.delete_prefix("-").delete_prefix("-").gsub("-", "_").to_sym }
                  .select { |k, _| VALID_INPUTS.include?(k) }

    new(options).run
  end

  def initialize(options)
    @opts = options.transform_keys(&:to_sym)

    @epochs = options[:epochs] || 10
    @mini_batch_size = options[:mini_batch_size] || 10
    @eta = options[:eta] || 0.25
    @lmbda = options[:lmbda] || 5.0

    @training_data = load_training_data
    @test_data = load_test_data unless options[:skip_tests]
  end

  def run
    output "Starting Stochastic Gradient Descent for Network #{network.layers}"

    network.sgd(
      training_data: training_data,
      epochs: epochs,
      mini_batch_size: mini_batch_size,
      eta: eta,
      lmbda: lmbda,
      test_data: test_data,
      **sgd_opts
    )
  end

  def network
    return @network if @network

    @network =
      if opts[:load_network]
        output "Loading network #{opts[:load_network]}"
        Network.load(opts[:load_network])
      elsif opts[:layers]
        output "Creating network with laters #{opts[:layers]}"
        Network.new(opts[:layers])
      else
        output "Creating default network with layers [784, 30, 10]"
        Network.new([784, 30, 10])
      end
  end

  def load_training_data
    output "Loading training data..."

    training_data = MnistLoader.load_training_data

    output "Training data loaded!"

    training_data
  end

  def load_test_data
    output "Loading test data..."

    test_data = MnistLoader.load_test_data

    output "Test data loaded!"

    test_data
  end

  def output(*args)
    puts "", *args, ""
  end

  def sgd_opts
    Network::VALID_SGD_OPTS.to_h { |key| [key, opts[key]] }
  end
end

CLI.run

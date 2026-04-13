require "csv"

require "numo/narray/alt"

module MnistLoader
  TRAINING_DATA_FILENAME = 'mnist_train.csv'
  TEST_DATA_FILENAME = 'mnist_test.csv'

  module_function

  def load_data(filename)
    CSV.foreach(filename, headers: true, converters: :numeric).map do |row|
      result = Numo::DFloat.zeros([10, 1])
      result[row[0]] = 1

      inputs = Numo::DFloat[row[1..]].reshape(784, 1)

      [inputs, result]
    end
  end

  def load_test_data
    load_data TEST_DATA_FILENAME
  end

  def load_training_data
    load_data TRAINING_DATA_FILENAME
  end
end
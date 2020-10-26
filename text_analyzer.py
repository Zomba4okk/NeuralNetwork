from typing import TYPE_CHECKING, List, Optional

import tensorflow as tf

from tools.data_preprocessing_tools import get_batch

if TYPE_CHECKING:
    from numpy import ndarray
    from sklearn.utils import Bunch


class TextAnalyzer:
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_1_size: int,
            hidden_2_size: int
    ) -> None:
        self.total_words_count = input_size
        self.categories_count = output_size
        self.input_tensor = tf.placeholder(tf.float32, [None, input_size], name='input')
        self.output_tensor = tf.placeholder(tf.float32, [None, output_size], name='output')
        self.weights = {
            'h1': tf.Variable(tf.random_normal([input_size, hidden_1_size])),
            'h2': tf.Variable(tf.random_normal([hidden_1_size, hidden_2_size])),
            'out': tf.Variable(tf.random_normal([hidden_2_size, output_size]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([hidden_1_size])),
            'b2': tf.Variable(tf.random_normal([hidden_2_size])),
            'out': tf.Variable(tf.random_normal([output_size]))
        }
        self.session = tf.Session()
        self.prediction = None

    def _multilayer_perceptron(self) -> tf.Tensor:
        layer_1_multiplication = tf.matmul(self.input_tensor, self.weights['h1'])
        layer_1_addition = tf.add(layer_1_multiplication, self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1_addition)

        # Hidden layer with RELU activation
        layer_2_multiplication = tf.matmul(layer_1, self.weights['h2'])
        layer_2_addition = tf.add(layer_2_multiplication, self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2_addition)

        # Output layer
        out_layer_multiplication = tf.matmul(layer_2, self.weights['out'])
        out_layer_addition = out_layer_multiplication + self.biases['out']

        return out_layer_addition

    def train_model(
            self,
            training_groups: 'Bunch',
            learning_rate: int,
            training_epochs: int,
            batch_size: int,
            display_step: int,
            test_groups: Optional['Bunch'] = None,
            path: Optional[str] = None
    ) -> None:
        # Construct model
        self.prediction = self._multilayer_perceptron()

        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.prediction,
            labels=self.output_tensor
        ))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # Initializing the variables
        init = tf.global_variables_initializer()

        self.session.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(training_groups.data) / batch_size)
            # Loop over all batches
            for index in range(total_batch):
                batch_x, batch_y = get_batch(
                    training_groups,
                    index,
                    batch_size,
                    self.total_words_count,
                    self.categories_count
                )
                # Run optimization op (backprop) and cost op (to get loss value)
                cost, _ = self.session.run(
                    [loss, optimizer],
                    feed_dict={self.input_tensor: batch_x, self.output_tensor: batch_y}
                )
                # Calculate average loss
                avg_cost += cost / total_batch

            # Display logs per epoch step
            if epoch % display_step == 0:
                print('Epoch: {epoch:04d}. Loss = {avg_cost:.9f};'.format(epoch=epoch + 1, avg_cost=avg_cost))

        print('Optimization Finished.')

        if test_groups:
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.output_tensor, 1))

            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            batch_x_test, batch_y_test = get_batch(
                test_groups,
                0,
                len(test_groups.target),
                self.total_words_count,
                self.categories_count
            )
            print(
                'Accuracy:',
                accuracy.eval({self.input_tensor: batch_x_test, self.output_tensor: batch_y_test}, session=self.session)
            )

        if path:
            # Save the variables to disk
            saver = tf.train.Saver()
            save_path = saver.save(self.session, path)

            print(f'Model saved in path: {save_path}')

    def load_model(self, path: str) -> None:
        saver = tf.train.Saver()

        saver.restore(self.session, path)
        print('Model restored.')

    def get_prediction(self, texts: 'ndarray') -> List[int]:
        return self.session.run(tf.argmax(self.prediction, 1), feed_dict={self.input_tensor: texts})


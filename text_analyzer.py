from typing import TYPE_CHECKING, List, Optional, Iterable

import tensorflow as tf

from constants import (
    batch_size,
    display_step,
    hidden_1_size,
    hidden_2_size,
    learning_rate,
    training_epochs,
)
from tools.data_preprocessing_tools import TextProcessingService

if TYPE_CHECKING:
    from numpy import ndarray
    from sklearn.utils import Bunch


class TextAnalyzer:
    hidden_1_size: int = hidden_1_size
    hidden_2_size: int = hidden_2_size
    learning_rate: int = learning_rate
    training_epochs: int = training_epochs
    batch_size: int = batch_size
    display_step: int = display_step

    def __init__(self, data_groups_list: Iterable['Bunch']) -> None:
        self.service = TextProcessingService(data_groups_list)

        self.input_tensor = tf.placeholder(tf.float32, [None, self.service.total_words_count], name='input')
        self.output_tensor = tf.placeholder(tf.float32, [None, self.service.categories_count], name='output')

        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.service.total_words_count, self.hidden_1_size])),
            'h2': tf.Variable(tf.random_normal([self.hidden_1_size, self.hidden_2_size])),
            'out': tf.Variable(tf.random_normal([self.hidden_2_size, self.service.categories_count]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.hidden_1_size])),
            'b2': tf.Variable(tf.random_normal([self.hidden_2_size])),
            'out': tf.Variable(tf.random_normal([self.service.categories_count]))
        }

        self.prediction = self._multilayer_perceptron()
        self.session = tf.Session()

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
            training_data_groups: 'Bunch',
            test_data_groups: Optional['Bunch'],
            path: Optional[str] = None
    ) -> None:
        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.prediction,
            labels=self.output_tensor
        ))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # Initializing the variables
        init = tf.global_variables_initializer()

        self.session.run(init)

        for epoch in range(self.training_epochs):
            avg_cost = 0.
            total_batch = int(len(training_data_groups.data) / self.batch_size)
            # Loop over all batches
            for index in range(total_batch):
                batch_x, batch_y = self.service.get_batch(training_data_groups, index, self.batch_size)
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

        if test_data_groups:
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.output_tensor, 1))

            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            batch_x_test, batch_y_test = self.service.get_batch(
                test_data_groups,
                0,
                len(test_data_groups.target)
            )
            print(
                'Accuracy:',
                accuracy.eval({self.input_tensor: batch_x_test, self.output_tensor: batch_y_test}, session=self.session)
            )

        if path:
            self.save_model(path)

    def save_model(self, path: str) -> None:
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


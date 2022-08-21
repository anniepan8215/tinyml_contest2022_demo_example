import tensorflow as tf
import keras.layers as nn
import keras
import keras.backend as K


class model_1_tf(tf.keras.Model):
    def __init__(self,training=True):
        super.__init__()
        self.save_hyperparameters()
        self.conv1 = tf.keras.models.Sequential(
            nn.Conv2D(1, 3, kernel_size=(6, 1), stride=(2, 1), padding=0, activation='relu'),
            nn.BatchNormalization(3, training=training,epsilon=1e-5, momentum=0.1)
        )

        self.conv2 = tf.keras.models.Sequential(
            nn.Conv2D(3, 5, kernel_size=(5, 1), stride=(2, 1), padding=0, activation='relu'),
            nn.BatchNormalization(5, training=training,epsilon=1e-5, momentum=0.1)
        )

        self.conv3 = tf.keras.models.Sequential(
            nn.Conv2D(5, 10, kernel_size=(4, 1), stride=(2, 1), padding=0, activation='relu'),
            nn.BatchNormalization(10, training=training,epsilon=1e-5, momentum=0.1)
        )

        self.conv4 = tf.keras.models.Sequential(
            nn.Conv2D(10, 20, kernel_size=(4, 1), stride=(2, 1), padding=0, activation='relu'),
            nn.BatchNormalization(20, training=training,epsilon=1e-5, momentum=0.1)
        )

        self.conv5 = tf.keras.models.Sequential(
            nn.Conv2D(20, 20, kernel_size=(4, 1), stride=(2, 1), padding=0, activation='relu'),
            nn.BatchNormalization(20,training=training, epsilon=1e-5, momentum=0.1)
        )

        self.fc1 = tf.keras.models.Sequential(
            nn.Dropout(0.5),
            nn.Dense(10, activation='relu')
        )
        self.fc2 = tf.keras.models.Sequential(
            nn.Dense(2)
        )

    def call(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1, 740)

        fc1_output = self.fc1(conv5_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output

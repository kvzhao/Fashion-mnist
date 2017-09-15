from __future__ import division
import os
import csv
import shutil
import gzip
import numpy as np
import tensorflow as tf
from convnet import ConvNet

# NETWORK PARAMETERS
tf.app.flags.DEFINE_integer('conv1_filters', 32, 'Number of filters of Convlutional layer')
tf.app.flags.DEFINE_integer('conv2_filters', 64, 'Number of filters of Convlutional layer')
tf.app.flags.DEFINE_integer('conv1_kernel', 5, 'Kernel size of Convlutional filters')
tf.app.flags.DEFINE_integer('conv2_kernel', 3, 'Kernel size of Convlutional filters')
tf.app.flags.DEFINE_integer('fc1_hiddens', 256, 'Hidden Units of Fully connected layer')

tf.app.flags.DEFINE_string('model_path', 'logs/CNN-Fashion', 'The path of trained model and checkpoints')
tf.app.flags.DEFINE_string('data_path', 'test_data', 'Path of assigned testing data path')

FLAGS = tf.app.flags.FLAGS

def load_model(sess, config):
    model = ConvNet(config, 'inference')
    model.build_model()
    if tf.train.checkpoint_exists(config.model_path):
        print ('Reloading model parameters...')
        model.restore(sess, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file: {}'.format(FLAGS.model_path)
        )
    return model

def inference():
    
    # Load the testing dataset
    images_path = os.path.join(FLAGS.data_path, '%s-images-idx3-ubyte.gz' % 't10k')
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(10000, 784)

    # Convert to float and normalize
    images = images.astype(np.float32)
    images /= 255

    with tf.Session() as sess:
        model = load_model(sess, FLAGS)
        labels = model.predict_step(sess, images)

        # Save the results to cvs
        with open('results.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'label'])
            for idx, label in enumerate(labels):
                writer.writerow([idx, label])
        
        print ('Save the inference outcomes to results.csv.')

def main():
    inference()

if __name__ == '__main__':
    main()
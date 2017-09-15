import os
import json
import shutil
import numpy as np
import tensorflow as tf
from convnet import ConvNet
from tensorflow.examples.tutorials.mnist import input_data

FASION_MNIST=True
if FASION_MNIST:
    data = input_data.read_data_sets('data/fashion-mnist', validation_size=5000)
else:
    data = input_data.read_data_sets('/tmp/mnist/input_data')
print ('Data set with {} training exmaples, {} testing and {} validation examples'.format(
    data.train.num_examples, data.test.num_examples, data.validation.num_examples))

# NETWORK PARAMETERS
tf.app.flags.DEFINE_integer('conv1_filters', 32, 'Number of filters of Convlutional layer')
tf.app.flags.DEFINE_integer('conv2_filters', 64, 'Number of filters of Convlutional layer')
tf.app.flags.DEFINE_integer('conv1_kernel', 5, 'Kernel size of Convlutional filters')
tf.app.flags.DEFINE_integer('conv2_kernel', 3, 'Kernel size of Convlutional filters')
tf.app.flags.DEFINE_integer('fc1_hiddens', 128, 'Hidden Units of Fully connected layer')

tf.app.flags.DEFINE_float('dropout', 0.5, 'Option of using dropout layer with ratio 0.5')
tf.app.flags.DEFINE_bool('batchnorm', False, 'Option of using batch normalization layer')

# TRAINING PROCESS
tf.app.flags.DEFINE_string('optimizer_type', 'SGD', 'Assign type of optimizers (SGD/AdaDelta/ADAM/RMSProp)')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Decay of learning rate')
tf.app.flags.DEFINE_integer('decay_steps', 15000, 'Learing rate decay after steps')
tf.app.flags.DEFINE_float('clip_grad', 5.0, 'Maximum of gradient norm')

tf.app.flags.DEFINE_integer('max_epochs', 10, 'Max epoch of training process')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 32, 'Batch size of testing data')

tf.app.flags.DEFINE_integer('save_freq', 10000, 'Save model per # of steps')
tf.app.flags.DEFINE_integer('display_freq', 100, 'Print step loss per # of steps')
tf.app.flags.DEFINE_integer('eval_freq', 500, 'Evaluate model per # of steps')

# General
#tf.app.flags.DEFINE_string('model_name', 'ConvNet', 'Name of the model')
tf.app.flags.DEFINE_string('logdir', 'logs', 'Name of output folder')
tf.app.flags.DEFINE_string('task_name', 'CNN-Fashion', 'Name of this training task')
tf.app.flags.DEFINE_bool('reset', True, 'Training start from stratch')

FLAGS = tf.app.flags.FLAGS

def create_model(sess, FLAGS):
    # TODO: model type?
    model = ConvNet(FLAGS, mode='train')
    model.build_model()

    # create log dir
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)
    # create task file
    model_path = '/'.join([FLAGS.logdir, FLAGS.task_name])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print ('Save model to {}'.format(model_path))
    elif (FLAGS.reset):
        shutil.rmtree(model_path)
        os.makedirs(model_path)
        print ('Remove existing model at {} and restart.'.format(model_path))

    # handle restore: if exist, restore or delete

    # initialize variables
    sess.run(tf.global_variables_initializer())

    return model

def train():

    with tf.Session() as sess:
        model = create_model(sess, FLAGS)
        total_batch = data.train.num_examples // FLAGS.batch_size
        print ('Run {} steps per Epoch. (Total Batch)'.format(total_batch))

        # summart writer
        model_path = '/'.join([FLAGS.logdir, FLAGS.task_name])
        log_writer = tf.summary.FileWriter(model_path, graph=sess.graph)

        loss_hist = [] 
        print ('Start Training...')
        for epoch_idx in range(FLAGS.max_epochs):
            epoch_loss = 0.0
            for step in range(total_batch):
                imgs, labels = data.train.next_batch(FLAGS.batch_size)
                step_loss, _ = model.train_step(sess, imgs, labels)
                epoch_loss += step_loss

                if model.global_step.eval() % FLAGS.display_freq == 0:
                    print ('GlobalSteps [ {} ]: training loss = {}'.format(model.global_step.eval(), step_loss))
                
                if model.global_step.eval() % FLAGS.eval_freq == 0:
                    test_imgs, test_labels = data.test.next_batch(FLAGS.test_batch_size)
                    eval_loss, accuracy, summary = model.eval_step(sess, test_imgs, test_labels)
                    # Only write evalidation results to summary
                    log_writer.add_summary(summary, model.global_step.eval())

                    print ('GlobalSteps [ {} ]: testing loss = {}, with Accuracy {}'.format(
                        model.global_step.eval(), step_loss, accuracy))

                if model.global_step.eval() % FLAGS.save_freq == 0:
                    print ('Saving the model...')
                    checkpoint_path = os.path.join(model_path, 'ConvNet')
                    model.save(sess, checkpoint_path, global_step=model.global_step)
                    # save the configuration
                    json.dump(model.config, open('{}-{}.json'.format(checkpoint_path, model.global_step.eval()), 'wb'), indent=2)

            epoch_loss /= total_batch 
            loss_hist.append(epoch_loss)
            print ('Epoch [ {} ]: Average losses {}'.format(epoch_idx, epoch_loss))

        print ('Minimum loss is {} @ {} epoch'.format(np.min(loss_hist), np.argmin(loss_hist)))

    print ('Saving the last model...')
    checkpoint_path = os.path.join(model_path, 'ConvNet')
    model.save(sess, checkpoint_path, global_step=model.global_step)
    # save the configuration
    json.dump(model.config, open('{}-{}.json'.format(checkpoint_path, model.global_step.eval()), 'wb'), indent=2)
    print ('Training terminated.')

def main():
    train()

if __name__ == '__main__':
    main()
import tensorflow as tf

class ConvNet(object):
    def __init__(self, config, mode):
        assert mode.lower() in ['train', 'inference']
        self.mode = mode.lower()
        self.config = config

        # hyper-params
        self.conv1_filters = config.conv1_filters
        self.conv1_kernel = config.conv1_kernel
        self.conv2_filters = config.conv2_filters
        self.conv2_kernel = config.conv2_kernel
        self.fc1_hiddens = config.fc1_hiddens

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # learning process
        if self.mode.lower() == 'train':
            self.dropout = config.dropout
            self.learning_rate = config.learning_rate
            self.decay_rate = config.decay_rate
            self.decay_steps = config.decay_steps
            self.clip_grad = config.clip_grad
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                            self.decay_steps, self.decay_rate, True)
            # strings
            self.optimizer_type = config.optimizer_type

            tf.summary.scalar('learning_rate', self.learning_rate)

    
    def build_model(self):
        print ('Start building model...')
        self._init_placeholders()
        self._build_convet()

        tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # summary 
        self.summary_op = tf.summary.merge_all()

        print ('Done.')
    
    def _init_placeholders(self):
        # image format: NHWC
        print ('\tInit placeholders')
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='inputs')
        if self.mode.lower() == 'train':
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, ], name='labels')
    
    def _build_convet(self):
        print ('\tInit ConvNet')
        x = tf.reshape(self.inputs, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(x,
                            filters=self.conv1_filters,
                            kernel_size=[self.conv1_kernel, self.conv1_kernel], 
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                            padding="same", activation=tf.nn.relu, name='Conv1')
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='Pool1')

        conv2 = tf.layers.conv2d(pool1, 
                            filters=self.conv2_filters,
                            kernel_size=[self.conv2_kernel, self.conv2_kernel], 
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                            padding="same", activation=tf.nn.relu, name='Conv2')
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='Pool2')

        flatten = tf.contrib.layers.flatten(pool2)

        fc1 = tf.layers.dense(flatten, self.fc1_hiddens, activation=tf.nn.relu, 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='FC1')
        if self.mode.lower() == 'train':
            if (self.dropout > 0.0):
                fc1 = tf.layers.dropout(fc1, rate=self.dropout, name='FC1Drop')

        self.logits = tf.layers.dense(fc1, units=10, name='Logits')
        self.preds = tf.argmax(input=self.logits, axis=1)

        # build loss function if in train mode
        if self.mode == 'train':
            print ('\tDefine loss')
            onehot_labels = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=10)
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=self.logits)
            
            tf.summary.scalar('loss', self.loss)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.preds, tf.argmax(onehot_labels, 1)), tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)
            
            self._init_optimizer()
    
    def _init_optimizer(self):
        print ('\tSetting optimizer')
        trainable_params = tf.trainable_variables()

        if self.optimizer_type.lower() == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_type.lower() == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_type.lower() == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_grad)

        self.updates = self.optimizer.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)
        
    def train_step(self, sess, input_batch, target_batch):
        if self.mode.lower() != 'train':
            raise ValueError ('Train step function can only used in train mode.')
        input_feed = {self.inputs: input_batch,
                        self.labels: target_batch}
        output_feed = [self.updates, self.loss, self.summary_op]
        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2] # loss and summary
    
    def eval_step(self, sess, input_batch, target_batch):
        # these data should come from test samples
        input_feed = {self.inputs: input_batch,
                        self.labels: target_batch}
        output_feed = [self.loss, self.accuracy, self.summary_op]
        outputs = sess.run(output_feed, input_feed)
        return outputs

    def predict_step(self, sess, input_batch):
        if self.mode == 'inference':
            input_feed = {self.inputs: input_batch}
            output_feed = [self.preds]
            output = sess.run(output_feed, input_feed)
            return output[0]

    def save(self, sess, path, var_list=None, global_step=None):
        saver = tf.train.Saver()
        save_path = saver.save(sess, path, global_step)
        print ('model (global steps: {})saved at {}'.format(global_step, save_path))

    def restore(self, sess, path):
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ('model restore from {}'.format(checkpoint.model_checkpoint_path))
        else:
            raise ValueError('Can not load from checkpoints')
        
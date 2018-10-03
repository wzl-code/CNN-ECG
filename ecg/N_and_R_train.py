from __future__ import print_function
import shutil
import os.path
import tensorflow as tf
import read2

EXPORT_DIR = 'D:/program/model'

if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)#递归删除一个目录以及目录内的所有内容

# Parameters
learning_rate = 0.001
training_iters = 3010
batch_size = 50
display_step = 10

# Network Parameters
n_classes = 2  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, shape=[None,128,128,1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create Model
def conv_net(x, weights, biases, dropout):

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([32 * 32 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#read data
image, label = read2.read_and_decode("N_and_R_train.tfrecords", batch_size)
# test
image_t, label_t = read2.read_and_decode("N_and_R_test.tfrecords", batch_size)
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 1
    # Keep training until reach max iterations
    #batch_size有128个，而x的placeholder为[None,784],应该是把数据堆叠起来了。
    try:
        while step * batch_size < training_iters:
            example, l = sess.run([image, label])
            #print(batch_x.shape)
            # Run optimization op (backprop)
            #一次给进[128，784]
            sess.run(optimizer, feed_dict={x: example, y: l,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: example,
                                                                  y: l,
                                                                  keep_prob: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

    #test
        example_t, l_t = sess.run([image_t, label_t])
        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={x: example_t,
                                            y: l_t,
                                            keep_prob: 1.}))
        WC1 = weights['wc1'].eval(sess)
        BC1 = biases['bc1'].eval(sess)
        WC2 = weights['wc2'].eval(sess)
        BC2 = biases['bc2'].eval(sess)
        WD1 = weights['wd1'].eval(sess)
        BD1 = biases['bd1'].eval(sess)
        W_OUT = weights['out'].eval(sess)
        B_OUT = biases['out'].eval(sess)

        # Create new graph for exporting
        g = tf.Graph()
        with g.as_default():
            x_2 = tf.placeholder("float", shape=[None,128,128,1], name="input")

            WC1 = tf.constant(WC1, name="WC1")
            BC1 = tf.constant(BC1, name="BC1")
            CONV1 = conv2d(x_2, WC1, BC1)
            MAXPOOL1 = maxpool2d(CONV1, k=2)

            WC2 = tf.constant(WC2, name="WC2")
            BC2 = tf.constant(BC2, name="BC2")
            CONV2 = conv2d(MAXPOOL1, WC2, BC2)
            MAXPOOL2 = maxpool2d(CONV2, k=2)

            WD1 = tf.constant(WD1, name="WD1")
            BD1 = tf.constant(BD1, name="BD1")

            FC1 = tf.reshape(MAXPOOL2, [-1, WD1.get_shape().as_list()[0]])
            FC1 = tf.add(tf.matmul(FC1, WD1), BD1)
            FC1 = tf.nn.relu(FC1)

            W_OUT = tf.constant(W_OUT, name="W_OUT")
            B_OUT = tf.constant(B_OUT, name="B_OUT")

            # skipped dropout for exported graph as there is no need for already calculated weights

            OUTPUT = tf.nn.softmax(tf.matmul(FC1, W_OUT) + B_OUT, name="output")

            sess = tf.Session()
            init = tf.initialize_all_variables()
            sess.run(init)

            graph_def = g.as_graph_def()
            tf.train.write_graph(graph_def, EXPORT_DIR, 'mnist_fang.pb', as_text=False)
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
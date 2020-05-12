'''


'''
import tensorflow as tf
from tensorflow .contrib import rnn

################################ Loading Data #################################
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


################################## parameters #################################
learning_rate = 0.001
batch_size = 128
iteration = 100000
disp_after = 20


n_input = 28
n_step = 28
n_units = 128
n_class = 10

################################## Variables ##################################
x = tf.placeholder("float" , [None , n_step , n_input])
y = tf.placeholder("float" , [None, n_class])

weights = {'out':tf.Variable(tf.random_normal([n_units,n_class]))}
biases = {'out':tf.Variable(tf.random_normal([n_class]))}

################################## NN Model ###################################
def RNN(x,weights,biases):
    
    x = tf.unstack(x,n_step,1)
    
    lstm_cell =  tf.contrib.rnn.GRUCell(n_units)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    e = tf.matmul(outputs[-1],weights['out'])+biases['out']
    return e
############################### Training Model ################################
pred= RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

################################ Runing Model##################################

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < iteration:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_step, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        
        if step % disp_after == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})           
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        
            
            
        step += 1
        
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_step, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
        




        

 
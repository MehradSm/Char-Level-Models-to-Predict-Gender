import tensorflow as tf
import numpy as np
import pickle


def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

############################ Model
def cnn_complete(x_, keep_prob_):
    pool_drop = x_
    kernel = []
    bias = []
    for i in range(len(k_size)):
        with tf.variable_scope("convolution"+str(i)):
            conv = tf.layers.conv1d(
                    inputs=pool_drop,
                    filters=n_filter[i],  # number of filters
                    kernel_size=k_size[i],
                    padding="same",
                    activation=tf.nn.relu,
                    name="conv")
            pool = tf.layers.max_pooling1d(inputs=conv, 
                                                pool_size=2, 
                                                strides=st_size[i],
                                                name="pool")  # convolution stride
            pool_drop = tf.nn.dropout(pool,keep_prob_[i], name='drop')
            tf.get_variable_scope().reuse_variables()
            kernel.append(tf.get_variable('conv/kernel'))
            bias.append(tf.get_variable('conv/bias'))
    with tf.variable_scope("fully_connected"):
        pool_flat = tf.contrib.layers.flatten(pool, scope='poolflat')
        dense = tf.layers.dense(inputs=pool_flat, units=500, activation=tf.nn.relu)
        dense_drop = tf.nn.dropout(dense, keep_prob_[i+1])
        #dense2 = tf.layers.dense(inputs=dense_drop, units=500, activation=tf.nn.relu)
        #dense_drop2 = tf.nn.dropout(dense2, keep_prob_[i+2])
        logits = tf.layers.dense(inputs=dense_drop, units=2)
        
    return [logits, kernel, bias]

############################ Inference
def Inference():
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.int32, [batch_size, maxlen])
            y_ = tf.placeholder(tf.int32, [batch_size])
            w_ = tf.placeholder(tf.float32, [batch_size])
            keep_prob_ = tf.placeholder(tf.float32)
            
            embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, x_)
            
            y_logits, kernel, bias = cnn_complete(embed,keep_prob_)  
            y_pred = tf.argmax(tf.nn.softmax(y_logits), dimension=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_sum(tf.multiply(w_,tf.cast(correct_prediction, tf.float32)))
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_logits)
            cross_entropy_loss = tf.reduce_sum(tf.multiply(w_,losses))
            #cross_entropy_loss = tf.reduce_mean(losses)       
            trainer = tf.train.AdamOptimizer()
            train_op = trainer.minimize(cross_entropy_loss)
        with tf.device("/cpu:0"):     
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
  #          image_shaped_input = tf.reshape(kernel[0], [-1, 5, 5, 3])
 #           tf.summary.image('kernel', image_shaped_input, 32)
            merge = tf.summary.merge_all()
            model_dict = {'train_op':train_op, 'accuracy':accuracy,
                          'loss':cross_entropy_loss, 'merge':merge,
                          'kernel':kernel, 'bias':bias, 
                          'inputs': [x_, y_, keep_prob_,w_],'graph': g,
                          'correct_prediction':correct_prediction}
    return model_dict

############################ Training

def Train(model_dict):
    with model_dict['graph'].as_default():
        sess = tf.InteractiveSession()
        #test_writer = tf.summary.FileWriter('summaries_dir/')
        sess.run(tf.global_variables_initializer())
        x_, y_, keep_prob_, w_ = model_dict['inputs']
        if change_arc == True:
            save_dict = {"convolution0/conv/kernel": model_dict['kernel'][0],
                         "convolution0/conv/bias": model_dict['bias'][0]}
            saver = tf.train.Saver(save_dict)
        else:
            saver = tf.train.Saver()
    
        if load_model:
            saver.restore(sess, load_point_name)
            print("Model restored.")
    
        for epoch_i in range(epoch_n):
            ids = np.arange(N_train)
            np.random.shuffle(ids)
            for batch_i in range(int(N_train/batch_size)):
                inter = range(batch_i*batch_size,(batch_i+1)*batch_size)
                feed = {x_: x_train[ids[inter]],
                                    y_: y_train[ids[inter]],
                                    keep_prob_: [0.7+0.3*epoch_i/epoch_n]*len(keep_prob),
                                    w_: w[ids[inter]]} ## dors kon
                to_compute = [model_dict['train_op'],model_dict['accuracy']]
                temp ,acc = sess.run(to_compute,feed_dict=feed)
                #print("   Training Accuracy: %.2f" %acc,end='\r')
                if batch_i % print_every == 0:
                    collect_arr = []
                    for test_batch_i in range(int(N_test/batch_size)):
                        feed = {x_: x_test[test_batch_i*batch_size:(test_batch_i+1)*batch_size,:],
                                    y_: y_test[test_batch_i*batch_size:(test_batch_i+1)*batch_size],
                                    keep_prob_: [1.0]*len(keep_prob),
                                    w_: [1/batch_size]*batch_size}
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, feed_dict = feed))
                        #summary = sess.run(model_dict['merge'], feed_dict = feed)
                        #test_writer.add_summary(summary, epoch_i*140+batch_i)
                    avgs = np.mean(collect_arr, axis=0)
                    print("epoch %d,  batch %d,  loss: %.3f,   accuracy: %.3f"
                          %(epoch_i,batch_i,avgs[0],avgs[1]))
                    
        if save_model:
            save_path = saver.save(sess, save_point_name +'/cnn'+str(run)+'.ckpt' )
            print("Model saved in file: %s" % save_path)
            
        I = np.array([],dtype = bool)
        for batch_i in range(int(N_train/batch_size)):
            feed = {x_: x_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                                y_: y_train[batch_i*batch_size:(batch_i+1)*batch_size],
                                keep_prob_: [1.0]*len(keep_prob),
                                w_:[1/batch_size]*batch_size}
            to_compute = [model_dict['correct_prediction']]
            I = np.append(I,sess.run(to_compute,feed_dict=feed))
        I_test = np.array([],dtype = bool)
        for batch_i in range(int(N_test/batch_size)):
            feed = {x_: x_test[batch_i*batch_size:(batch_i+1)*batch_size,:],
                                y_: y_test[batch_i*batch_size:(batch_i+1)*batch_size],
                                keep_prob_: [1.0]*len(keep_prob),
                                w_:[1/batch_size]*batch_size}
            to_compute = [model_dict['correct_prediction']]
            I_test = np.append(I_test,sess.run(to_compute,feed_dict=feed))
    return I , I_test

############################# Inputs 
# Data

x_train = load_obj('X_train') + 1
y_train = load_obj('y_train')
x_test = load_obj('X_test') + 1
y_test = load_obj('y_test')
#y = np.array(y[:,0,1:5],dtype = np.float16)
y_train = np.array(y_train[:,0,0],dtype = np.int8)
y_test = np.array(y_test[:,0,0],dtype = np.int8)

## Equalizing
I_f = np.where(y_train == 1)[0]
I_m = np.where(y_train == 0)[0]
I_f = I_f[0:len(I_m)]
x_train = np.append(x_train[I_f] , x_train[I_m], axis = 0)
y_train = np.append(y_train[I_f] , y_train[I_m], axis = 0)
del I_f , I_m

## Equalizing
I_f = np.where(y_test == 1)[0]
I_m = np.where(y_test == 0)[0]
I_f = I_f[0:len(I_m)]
x_test = np.append(x_test[I_f] , x_test[I_m], axis = 0)
y_test = np.append(y_test[I_f] , y_test[I_m], axis = 0)
del I_f , I_m

[N_train,maxlen] = x_train.shape
[N_test,maxlen] = x_test.shape
batch_size = 256
N_train = batch_size*int(N_train/batch_size)
N_test = batch_size*int(N_test/batch_size)
#N_train = 5888
x_train = x_train[:N_train]
y_train = y_train[:N_train]
x_test = x_test[:N_test]
y_test = y_test[:N_test]

# new
vocabulary_size = 60
embedding_size = 16

# Setting
epoch_n = 4
print_every = 1024
load_model = False
save_model = True
change_arc = False
save_point_name = "checkpoint"
load_point_name = "checkpoint/cnn.ckpt"
keep_prob = [0.7]*5
n_filter = [32, 64, 64,64]
k_size = [3,3,3,3]
st_size = [2,2,2,2]

N_boost = 10
w = np.array([1/batch_size]*N_train)
G = 0*y_train
G_test = 0*y_test
for run in range(N_boost):
    model_dict = Inference()
    I , I_test= Train(model_dict)
    I = ~I
    I_test = ~I_test
    err = batch_size*np.dot(w,I)/N_train
    alpha = np.log((1-err)/err)
    w = w*np.exp(alpha*I)
    w = w*(N_train/batch_size)/np.sum(w)
    tot_train_error = np.mean(I)
    G = G + alpha*(2*I-1)
    G_test = G_test + alpha*(2*I_test-1)
    
    print("Alpha: %.5f "%alpha)
    print("Importance Error: %.5f "%err)
    print("Training Error of this run: %.5f"%tot_train_error)
    print("Training Error of boosted Classifier: %.6f"%np.mean(G>=0))
    print("Test Error of boosted Classifier: %.6f"%np.mean(G_test>=0))
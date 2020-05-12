import tensorflow as tf
import numpy as np
import pickle


def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

################################# Setting #####################################  

epoch_n = 8
batch_size = 64
print_every = 1024
load_model = False
save_model = True
change_arc = False
save_point_name = "checkpoint"
load_point_name = "checkpoint/rnn.ckpt"
shape = dict(n_steps_per_batch=1024, n_unique_ids=2, n_hidden_dim=200,
             embedding_size = 16,vocabulary_size=60,number_of_layers=1)
    
################################# Data ########################################

# Loading Data

x_train = load_obj('X_train') + 1
y_train = load_obj('y_train')

x_test = load_obj('X_test') + 1
y_test = load_obj('y_test')

y_train = np.array(y_train[:,0,0],dtype = np.int8)
y_test = np.array(y_test[:,0,0],dtype = np.int8)

## Equalizing Training Data

I_f = np.where(y_train == 1)[0]
I_m = np.where(y_train == 0)[0]
I_f = I_f[0:len(I_m)]
x_train = np.append(x_train[I_f] , x_train[I_m], axis = 0)
y_train = np.append(y_train[I_f] , y_train[I_m], axis = 0)
del I_f , I_m


## Equalizing Test Data
I_f = np.where(y_test == 1)[0]
I_m = np.where(y_test == 0)[0]
I_f = I_f[0:len(I_m)]
x_test = np.append(x_test[I_f] , x_test[I_m], axis = 0)
y_test = np.append(y_test[I_f] , y_test[I_m], axis = 0)
del I_f , I_m

[N_train,maxlen] = x_train.shape
[N_test,maxlen] = x_test.shape
N_train = batch_size*int(N_train/batch_size)
N_test = batch_size*int(N_test/batch_size)

x_train = x_train[:N_train]
y_train = y_train[:N_train]
x_test = x_test[:N_test]
y_test = y_test[:N_test]


def get_default_gpu_session(fraction=0.333):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    return tf.Session(config=config)    


################################### Model #####################################
def RNN(x_,keep_prob):
    
    x = tf.unstack(x_, shape['n_steps_per_batch'], 1)
    
    with tf.variable_scope('weights',reuse=True):  
        w_y = tf.get_variable('W_y', [shape['n_hidden_dim'] , shape['n_unique_ids']])
        b_y = tf.get_variable('b_y',  shape['n_unique_ids'])
        
    multi_lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=shape['n_hidden_dim'],forget_bias=1.0,
                                          layer_norm=True,dropout_keep_prob=keep_prob)
  
    
    outputs, states = tf.contrib.rnn.static_rnn(multi_lstm, x, dtype=tf.float32)
            
    logits = tf.matmul(outputs[-1],w_y) + b_y

    return logits


################################ Inference ####################################
def Inference():
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.int32, [batch_size, maxlen])
            y_ = tf.placeholder(tf.int32, [batch_size])
            keep_prob = tf.placeholder(tf.float32)
            
            
            embeddings = tf.Variable(
            tf.random_uniform([shape['vocabulary_size'], shape['embedding_size']], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, x_)
            tf.shape(embed)
            y_logits = RNN(embed,keep_prob)  
            y_pred = tf.argmax(y_logits, dimension=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_logits)
            cross_entropy_loss = tf.reduce_mean(losses)       
            trainer = tf.train.AdamOptimizer()
            train_op = trainer.minimize(cross_entropy_loss)
        with tf.device("/cpu:0"):     
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
            merge = tf.summary.merge_all()
            model_dict = {'train_op':train_op, 'accuracy':accuracy,
                          'loss':cross_entropy_loss, 'merge':merge,
                          'inputs': [x_, y_,keep_prob],'graph': g}
    return model_dict

################################## Training ###################################

def Train(model_dict):
    with model_dict['graph'].as_default():
        #sess = tf.InteractiveSession()
        sess = get_default_gpu_session(fraction=0.333)
        test_writer = tf.summary.FileWriter('summaries_dir/')
        sess.run(tf.global_variables_initializer())
        x_, y_,keep_prob= model_dict['inputs']
        saver = tf.train.Saver()

        if load_model:
            saver.restore(sess, load_point_name)
            print("Model restored.")
    
        for epoch_i in range(epoch_n):
            ids = np.arange(N_train)
            np.random.shuffle(ids)
            for batch_i in range(int(N_train/batch_size)):
                inter = range(batch_i*batch_size,(batch_i+1)*batch_size)
                feed = {x_: x_train[ids[inter]],y_: y_train[ids[inter]], keep_prob:0.7} 
                to_compute = [model_dict['train_op'],model_dict['accuracy']]
                _ ,acc = sess.run(to_compute,feed_dict=feed)
                if batch_i % print_every == 0:
                    collect_arr = []
                    for test_batch_i in range(int(N_test/batch_size)):
                        inter = range(test_batch_i*batch_size,(test_batch_i+1)*batch_size)
                        feed = {x_: x_test[inter], y_: y_test[inter],keep_prob:1.0}
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, feed_dict = feed))
                        summary = sess.run(model_dict['merge'], feed_dict = feed)
                        test_writer.add_summary(summary, epoch_i*140+batch_i)
                    avgs = np.mean(collect_arr, axis=0) #mean loss and acc
                    print("epoch %d,  batch %d,  loss: %.3f,   accuracy: %.3f"
                          %(epoch_i,batch_i,avgs[0],avgs[1]))
                    
        if save_model:
            save_path = saver.save(sess, save_point_name +'/rnn.ckpt' )
            print("Model saved in file: %s" % save_path)
        arr = []    
        for batch_i in range(int(N_train/batch_size)):
            
            inter = range(batch_i*batch_size,(batch_i+1)*batch_size)
            feed = {x_: x_train[inter],y_: y_train[inter]} 
            to_compute = [model_dict['loss'], model_dict['accuracy']]
            arr.append(sess.run(to_compute, feed_dict = feed))
        
        avgs = np.mean(arr, axis=0) #mean loss and acc
        print("training_loss %.3f , training_acc %.3f" % (avgs[0],avgs[1]))
            




model_dict = Inference()
Train(model_dict)

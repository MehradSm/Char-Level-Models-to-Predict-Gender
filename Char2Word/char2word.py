import numpy as np
import pickle


def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

############################# Inputs 
# Data

x_train = load_obj('X_test')
y_train = load_obj('y_test')
x_test = load_obj('X_test')
y_test = load_obj('y_test')
#y = np.array(y[:,0,1:5],dtype = np.float16)
y_train = np.array(y_train[:,0,0],dtype = np.int8)
y_test = np.array(y_test[:,0,0],dtype = np.int8)



##
[N_train,maxlen] = x_train.shape
[N_test,maxlen] = x_test.shape
vocab_size = 59

puncs = list(range(31)) + [57,58]
puncs.remove(3)


word_dict = {}
word_count = []
key = 0
for i in range(1000):
    temp = []
    for j in range(maxlen):
        if x_test[i,j] in puncs:
            if temp not in word_dict.values():
                word_dict.update({key:temp})
                temp = []
                key += 1
        else:
            temp.append(x_test[i,j])

chars = np.array(load_obj('chars'))                    
#chars = load_obj('chars')
#char_indices = dict((c, i) for i, c in enumerate(chars))
#indices_char = dict((i, c) for i, c in enumerate(chars))

for i in range(1000):
    print(''.join(chars[word_dict[i]]) )
    
    
string = "I am that I am"
my_string = string.lower().split()
my_dict = {}
for item in my_string:
  my_dict[item] = my_string.count(item)
print(my_dict)


def word_count(string):
    my_string = string.lower().split()
    my_dict = {}
    for item in my_string:
        if item in my_dict:
            my_dict[item] += 1
        else:
            my_dict[item] = 1
    print(my_dict)


import numpy as np
import pickle


def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

############################# Inputs 
# Data

x_train = load_obj('X_train')
y_train = load_obj('y_train')
x_test = load_obj('X_test')
y_test = load_obj('y_test')
#y = np.array(y[:,0,1:5],dtype = np.float16)
y_train = np.array(y_train[:,0,0],dtype = np.int8)
y_test = np.array(y_test[:,0,0],dtype = np.int8)

## Equlizing

I_f = np.where(y_test == 1)[0]
I_m = np.where(y_test == 0)[0]
I_f = I_f[0:len(I_m)]
#x_test = x_test[I_f]
#y_test = y_test[I_f]
x_test = np.append(x_test[I_f] , x_test[I_m], axis = 0)
y_test = np.append(y_test[I_f] , y_test[I_m], axis = 0)


##
[N_train,maxlen] = x_train.shape
[N_test,maxlen] = x_test.shape
vocab_size = 59

nf = np.sum(y_train)
nm = N_train - np.sum(y_train)
phi_f = nf/(nm+nf)

pf = np.zeros(shape = vocab_size)
pm = np.zeros(shape = vocab_size)

for i in range(vocab_size):
    pf[i] = np.sum(y_train*np.sum(x_train==i, axis = 1))
    pm[i] = np.sum((1-y_train)*np.sum(x_train==i, axis = 1))

pf = np.append (pf,1)
pm = np.append (pm,1)

#I = np.array([59],dtype = int)
I = [59]
for ndim in range(59):
    print("Step: ",ndim)
    maxacc = 0
    for j in range(59):
        I.append(j)
        pf2 = np.copy(pf/(np.sum(pf)-np.sum(pf[I])))
        pm2 = np.copy(pm/(np.sum(pm)-np.sum(pm[I])))
        pf2[I] = 1
        pm2[I] = 1
        y_pred = np.sum(np.log(pf2[x_train]),axis = 1)  > np.sum(np.log(pm2[x_train]),axis = 1)
        acc = np.mean(y_pred == y_train)
        del(I[-1])
        if acc > maxacc:
            if j not in I:
                print(acc,j)
                imax = j
                maxacc = acc
    I.append(imax)

print("These are the least important indicies")
for i in I:
    print(i)    
pf2 = np.copy(pf/(np.sum(pf)-np.sum(pf[I])))
pm2 = np.copy(pm/(np.sum(pm)-np.sum(pm[I])))
pf2[I] = 1
pm2[I] = 1     
y_pred_test = np.sum(np.log(pf2[x_test]),axis = 1) > np.sum(np.log(pm2[x_test]),axis = 1)
y_pred = np.sum(np.log(pf2[x_train]),axis = 1)  > np.sum(np.log(pm2[x_train]),axis = 1)

non_empty = maxlen- np.sum(x_train == -1, axis =1)
print("Accuracy: ",np.mean(y_pred == y_train))
print("Test Accuracy: ",np.mean(y_pred_test == y_test))
print("mean status length of females: ",np.sum(y_train*non_empty)/nf)
print("mean status length of males: ",np.sum((1-y_train)*non_empty)/nm)

    

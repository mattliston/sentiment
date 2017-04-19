# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python dnn.py
from six.moves import cPickle
import numpy as np
import theano
import theano.tensor as T
import lasagne as L
import random
import time

d=open('data.pickle','rb') #load data and labels
data=np.asarray(cPickle.load(d))
data=data.astype(np.float32)
d.close()
l=open('label.pickle','rb')
label=np.asarray(cPickle.load(l))
label=label.astype(np.int32)
l.close()

for i in range(0,label.size):
    if label[i]==-1:
        label[i]=2
dist=np.zeros(3)
for i in range(0,label.size):
    if label[i]==2: #this is negative change
        dist[0]+=1
    if label[i]==0:
        dist[1]+=1
    if label[i]==1:
        dist[2]+=1

print 'dist', dist

train_split=0.8 #break into train and test sets
train_data=data[0:int(train_split*data.shape[0])]
train_label=label[0:int(train_split*label.shape[0])]
test_data=data[int(train_split*data.shape[0]):data.shape[0]]
test_label=label[int(train_split*label.shape[0]):label.shape[0]]

train_dist=np.zeros(3)
for i in range(0,train_label.shape[0]):
    if train_label[i]==2:
        train_dist[0]+=1
    if train_label[i]==0:
        train_dist[1]+=1
    if train_label[i]==1:
        train_dist[2]+=1

test_dist=np.zeros(3)

for i in range(0,test_label.shape[0]):
    if test_label[i]==2:
        test_dist[0]+=1
    if test_label[i]==0:
        test_dist[1]+=1
    if test_label[i]==1:
        test_dist[2]+=1

#exit(0)
input_var = T.matrix(dtype=theano.config.floatX) #define network
target_var = T.vector(dtype='int32')
network = L.layers.InputLayer((None, data.shape[1]), input_var)
network = L.layers.DenseLayer(network, 100)
network = L.layers.DenseLayer(network, 50)
network = L.layers.DenseLayer(network, 25)
#network = L.layers.DenseLayer(network, 12)
#network = L.layers.DenseLayer(network, 11)
network = L.layers.DenseLayer(network, 3, nonlinearity=L.nonlinearities.softmax)
prediction = L.layers.get_output(network)
loss = L.objectives.aggregate(L.objectives.categorical_crossentropy(prediction, target_var), mode='mean')
params = L.layers.get_all_params(network, trainable=True)
updates = L.updates.adam(loss, params, learning_rate=0.000001)
scaled_grads,norm = L.updates.total_norm_constraint(T.grad(loss,params), np.inf, return_norm=True)
train_fn = theano.function([input_var, target_var], [loss,norm], updates=updates)
test_fn = theano.function([input_var], L.layers.get_output(network, deterministic=True))

for epoch in range(0,100):
    rng_state = np.random.get_state()
    np.random.shuffle(train_data)
    np.random.set_state(rng_state)
    np.random.shuffle(train_label)
    n_loss = 0
    t_loss = 0
    t_norm = 0
    batch_size=10
    for i in range(0,train_data.shape[0]-batch_size,batch_size): 
        b_loss,b_norm = train_fn(train_data[i:i+batch_size],train_label[i:i+batch_size])
        t_loss += b_loss
        t_norm += b_norm
        n_loss += 1.0        
    correct=0
    total=0
    hist=np.zeros(3)
    for i in range(0,test_data.shape[0]):
#        print test_data.shape[0]
        val_output = test_fn([test_data[i]])
        val_predictions = np.argmax(val_output[0])
#        print 'val_predictions', val_predictions
        if val_predictions==2:
            hist[0]+=1
        if val_predictions==0:
            hist[1]+=1
        if val_predictions==1:
            hist[2]+=1
#        hist[val_predictions]+=1
#        print val_predictions
#        print test_label[i]
#        exit(0)
        if val_predictions==test_label[i]:
            correct+=1
            total+=1
        else:
            total+=1
    tacc=float(correct)/float(total)

    print 'epoch', epoch, 't_loss', t_loss/n_loss, 't_norm', t_norm/n_loss, 'tacc', tacc, 'hist', hist,'train dist',train_dist, 'testdist', test_dist
    f = open('model.pickle', 'wb')
    cPickle.dump(L.layers.get_all_param_values(network), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()



#for i in range(0,len(data)):
#    print data[i], label[i]
#
#print data.shape, label.shape

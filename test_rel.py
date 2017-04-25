# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python test_rel.py
from six.moves import cPickle
import numpy as np
import theano
import theano.tensor as T
import lasagne as L
import random
import time
import string
from scipy.cluster.vq import kmeans,vq, kmeans2, whiten
import argparse
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=200)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})

# parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--k', help='kmeans clusters', default=10, type=int)
parser.add_argument('--minit', help='kmeans initialization method, random or points',default='points')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

f = open('model.pickle', 'rb')
[train_data,train_label,test_data,test_label,params] = cPickle.load(f)
f.close()

input_var = T.tensor3(dtype=theano.config.floatX) #define network
target_var = T.vector(dtype='int32')
network = L.layers.InputLayer((None,1,99),input_var)
network = L.layers.Conv1DLayer(network,num_filters=75,filter_size=10,stride=9)
network = L.layers.Conv1DLayer(network,num_filters=75,filter_size=10,stride=1)
#network = L.layers.MaxPool1DLayer(network,pool_size=10)
network = L.layers.DenseLayer(network, 50)
network = L.layers.DenseLayer(network, 25)
network = L.layers.DenseLayer(network, 12)
featvec=network
#network = L.layers.DenseLayer(network, 11)
network = L.layers.DenseLayer(network, 3, nonlinearity=L.nonlinearities.softmax)
print L.layers.get_output_shape(network)
test_fn = theano.function([input_var], [L.layers.get_output(network, deterministic=True),L.layers.get_output(featvec, deterministic=True)])

L.layers.set_all_param_values(network, params)

#maps each number to a character
cd={}
j=1
for c in string.printable:
    cd[j]=c 
    j+=1
cd[10001]=' '
cd[10000]='_'

headlines=[]
for i in range(0,test_data.shape[0]):
    s=''
    for j in range(test_data.shape[1]):
        s += cd[test_data[i,j]]
    headlines.append(s)

fv=[]
for i in range(0,test_data.shape[0]):
    reshape=np.reshape(test_data[i],(1,1,99))
    val_output,feat = test_fn(reshape)
    fv.append(feat[0])
fv = np.asarray(fv)
print 'fv.shape',fv.shape
centroids,cluster = kmeans2(fv, args.k, minit=args.minit)
order = np.argsort(cluster)
print 'centroids.shape', centroids.shape, 'cluster.shape', cluster.shape, 'order.shape',order.shape

predhist=np.zeros((args.k,3))
labelhist=np.zeros((args.k,3))
correcthist=np.zeros(args.k)
totalhist=np.zeros(args.k)
correct=0
total=0
for i in range(0,test_data.shape[0]):
    reshape=np.reshape(test_data[i],(1,1,99))
    val_output,feat = test_fn(reshape)
    val_predictions = np.argmax(val_output[0])
    predhist[cluster[i],val_predictions]+=1
    labelhist[cluster[i],test_label[i]]+=1
    if val_predictions==test_label[i]:
        correcthist[cluster[i]]+=1
        totalhist[cluster[i]]+=1
        correct+=1
        total+=1
    else:
        totalhist[cluster[i]]+=1
        total+=1

for i in range(0,test_data.shape[0]):
    ir = order[i]
    reshape=np.reshape(test_data[ir],(1,1,99))
    val_output,feat = test_fn(reshape)
    val_predictions = np.argmax(val_output[0])
    print 'cluster {:4d} prediction {:4d} label {:4d} headline {}'.format(cluster[ir],val_predictions,test_label[ir],headlines[ir])

print 'predhist',predhist
print 'labelhist',labelhist
print 'accuracyhist',correcthist/totalhist

tacc=float(correct)/float(total)
print 'tacc', tacc


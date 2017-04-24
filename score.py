# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python score.py
import argparse
import csv
from bs4 import BeautifulSoup
import re
import urllib2
from datetime import timedelta
from datetime import datetime
from six.moves import cPickle
import numpy as np
import string
import theano
import theano.tensor as T
import lasagne as L
import random
parser=argparse.ArgumentParser()
parser.add_argument('--weekend',help='add if weekend',default=False,type=bool)
args = parser.parse_args()
w=[0.56,0.38,0.50,0.36,0.17,0.04,0.32,0.05,0.04,0.23,0.05,0.15,0.09,0.15,0.05,0.05,0.06,0.13,0.04,0.39,0.07,0.04,0.15,1.27,1.23,0.70,1.77,0.07,0.11,0.17,0.28,0.27,0.27,0.07,0.10,0.07,0.06,0.59,0.11,0.16,0.14,0.22,0.16,0.09,0.03,3.68,0.21,0.13,0.06,0.05,0.03,1.23,0.09,0.23,0.01,0.10,0.13,0.04,0.12,0.06,1.14,0.09,0.13,0.17,0.19,0.03,1.54,0.07,0.29,0.23,0.02,0.51,0.04,0.10,0.17,0.44,0.43,0.04,0.05,0.05,0.06,0.06,0.20,0.11,0.05,0.12,0.27,0.04,0.05,0.13,0.48,0.06,0.06,0.07,0.09,0.03,0.23,0.36,0.02,0.98,0.07,0.32,0.06,0.19,0.05,0.05,0.05,0.81,0.80,0.09,0.06,0.20,0.06,0.05,0.83,0.18,0.32,0.89,0.06,0.08,0.09,0.30,0.12,0.15,0.12,0.37,0.04,0.17,0.02,0.22,0.11,0.40,0.06,0.26,0.05,0.05,0.16,0.10,0.16,0.07,0.10,0.09,0.12,0.02,0.03,0.09,0.09,0.24,0.06,0.34,0.09,0.09,0.34,0.29,0.05,0.06,0.17,0.17,0.16,0.13,0.10,0.14,0.19,0.07,0.03,0.27,0.05,0.08,0.16,0.12,0.08,0.09,0.09,0.16,0.08,0.05,0.20,0.05,1.66,0.04,1.67,0.06,0.05,0.22,0.13,0.09,0.07,0.12,0.02,0.03,0.03,0.05,0.07,0.05,0.22,0.09,0.05,0.07,0.08,0.03,0.03,0.27,1.30,0.07,0.17,0.23,0.07,0.43,0.06,0.40,0.04,0.05,0.20,0.04,0.05,0.07,0.09,0.05,0.12,0.07,0.03,0.07,0.06,0.15,0.06,0.89,0.46,0.05,0.07,0.16,0.16,0.07,0.07,0.21,0.13,0.11,0.11,0.85,0.18,0.69,0.11,0.05,0.05,0.14,0.15,0.06,0.04,0.04,0.03,0.08,1.64,0.19,1.50,0.05,0.05,0.09,0.09,0.24,0.05,0.20,0.08,0.03,0.27,0.14,0.06,0.06,0.07,0.11,0.03,0.05,0.04,0.09,0.39,0.07,0.04,0.34,0.06,0.35,0.14,0.12,0.04,0.04,0.02,0.06,0.13,0.14,0.19,0.07,0.05,0.53,0.04,0.06,0.54,0.14,0.08,0.55,0.86,0.28,0.06,0.03,0.08,0.14,2.50,0.06,0.07,0.09,0.34,0.25,0.09,0.10,0.30,0.07,0.02,0.08,0.04,0.07,0.02,0.05,0.30,0.11,0.03,0.09,0.02,0.01,0.31,0.07,0.37,0.04,0.07,0.03,0.16,0.09,0.21,0.03,0.09,0.29,0.12,0.24,0.10,0.06,0.66,0.11,0.10,0.02,0.09,0.24,0.05,0.03,0.81,0.03,0.05,1.00,0.17,0.88,0.17,0.05,0.15,0.28,0.02,0.31,0.13,0.17,0.43,0.08,1.14,0.11,0.14,0.22,0.11,0.17,0.03,0.04,0.04,0.39,0.03,0.07,0.03,0.22,0.08,0.08,0.05,0.14,0.08,0.07,0.26,0.03,0.10,0.08,0.10,0.13,0.08,0.02,0.17,0.27,0.05,0.53,0.04,0.07,0.04,0.14,0.13,0.02,0.28,0.09,0.05,0.05,0.25,0.17,0.10,0.03,0.43,0.14,0.03,0.20,0.13,0.09,0.13,0.13,0.09,0.15,0.13,0.03,0.02,0.04,0.40,0.06,0.24,0.09,0.05,0.08,0.05,0.17,0.82,0.30,0.05,0.39,0.25,0.04,0.04,0.04,0.06,0.02,0.02,0.16,0.07,0.09,0.39,0.05,0.08,0.02,0.02,0.43,0.10,0.81,0.36,0.05,0.42,0.05,0.05,0.09,0.14,0.04,0.12,0.04,0.06,0.99,0.14,0.08,0.83,0.09,0.08,0.56,0.36,0.15,0.06,0.09,1.17,0.13,0.12,0.05,0.07,0.13,0.06,0.06,0.12,0.08,0.05,0.05,0.11,0.03,0.07,0.05,0.04,0.20,0.11,0.12,0.04,0.13,0.0]

i=0
weight_map={}
with open('constituents.csv') as csvfile:
    reader=csv.reader(csvfile,delimiter=',')
    for row in reader:
        weight_map[row[0]]=w[i]
        i=i+1
print weight_map['SPY']

urls=[1]
data=[]
for t in weight_map: #load raw data
    for p in urls:
        try:
            url='https://www.benzinga.com/stock-articles/'+t+'/all'
            print 'url',url
            html = urllib2.urlopen(url).read()
            soup = BeautifulSoup(html)
            a = soup.find_all('div',class_='views-field-title')
            for b in a:
                date = re.findall(r' (.*?)</h|$',str(b.parent.parent.parent.h3))[0]
                article = re.findall(r'">(.*?)</a>|$',str(b.span.a))[0]
                if args.weekend:
                
#                print str(datetime.strptime(date,'%B %d, %Y').date()), str(datetime.today().date()) 
                    if (str(datetime.strptime(date,'%B %d, %Y').date())==str(datetime.today().date())) or (str(datetime.strptime(date,'%B %d, %Y').date())==str(datetime.today().date()-timedelta(days=1))) or (str(datetime.strptime(date,'%B %d, %Y').date())==str(datetime.today().date()-timedelta(days=2))):
                        if [t,date,article] not in data:
                            data.append([t,str(datetime.strptime(date,'%B %d, %Y').date()),article])
                            print t,str(datetime.strptime(date,'%B %d, %Y').date()),article
                    else:
                        continue
  
                else:

                    if (str(datetime.strptime(date,'%B %d, %Y').date())==datetime.today().date()):
                        data.append([t,str(datetime.strptime(date,'%B %d, %Y').date()),article])
                    else:
                        continue
        except urllib2.HTTPError:
            continue

    try:
        url='https://www.benzinga.com/stock/'+t.lower()
        print 'url',url
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html)
        st=soup.find_all('li',class_='story')
        bt=soup.find_all('span',class_='date')
        articles=[]
        date_2=[]
        for p in st:
           # print p
           x=re.findall(r'">(.*?)-0|$',str(p))
           for i in x:
               b=re.findall(r'(.*?)</a',str(i))
#               c=re.findall(r' (.*?)7 ',str(i))
               if b:
                   articles.append(b[0])
                #   print b
        for j in bt:
    #        print j
            ass=re.findall(r'">(.*?)-0400|$',str(j))[0]
            if ass:

                date_2.append(ass[0:-1])
               # print t
        for i in range(0,len(date_2)):
            if args.weekend:
#                print date_2[i], type(date_2[i])
#                print str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date())
                if (str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date())==str(datetime.today().date())) or (str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date())==str(datetime.today().date()-timedelta(days=1))) or (str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date())==str(datetime.today().date()-timedelta(days=2))): 
                    if [t,str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date()),articles[i]] not in data:
                        data.append([t,str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date()),articles[i]])
                        print t,str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date()),articles[i]
            else:       
  #add for normal days

                if (str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date())==str(datetime.today().date())):
                    if [t,str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date()),articles[i]] not in data:
                        data.append([t,str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date()),articles[i]])
                        print t,str(datetime.strptime(date_2[i],'%a, %d %b %Y %X').date()),articles[i]
                  #  data.append([t,date_2[i],articles])
                else:
                    continue
    except urllib2.HTTPError:
        continue

print len(data)

character_dict={} #encode headlines
j=1
for c in string.printable:
    character_dict[c]=j
    j+=1

x=[]

for i in range(0,len(data)): #encodes data
    temp=[]
    j=0
    for c in range(0,99):
        if c>=len(data[i][2]):
            temp.append(10001) #end of headline indicator
        else:
            try:
                temp.append(character_dict[data[i][2][c]])
                j+=1
            except KeyError:
                temp.append(10000)
    x.append(temp)

windows = np.asarray(x) #windows to be predicted
windows=windows.astype(np.float32)

underlying_score=0
spy_score=0
previous='MMM'
current_total=0
count=0

input_var = T.tensor3(dtype=theano.config.floatX) #define network
target_var = T.vector(dtype='int32')
network = L.layers.InputLayer((None,1,99),input_var)
network = L.layers.Conv1DLayer(network,num_filters=75,filter_size=10,stride=9)
network = L.layers.Conv1DLayer(network,num_filters=75,filter_size=10,stride=1)
#network = L.layers.MaxPool1DLayer(network,pool_size=10)
network = L.layers.DenseLayer(network, 50)
network = L.layers.DenseLayer(network, 25)
network = L.layers.DenseLayer(network, 12)
#network = L.layers.DenseLayer(network, 11)
network = L.layers.DenseLayer(network, 3, nonlinearity=L.nonlinearities.softmax)
prediction = L.layers.get_output(network)
loss = L.objectives.aggregate(L.objectives.categorical_crossentropy(prediction, target_var), mode='mean')
params = L.layers.get_all_params(network, trainable=True)
updates = L.updates.adam(loss, params, learning_rate=0.0001)
scaled_grads,norm = L.updates.total_norm_constraint(T.grad(loss,params), np.inf, return_norm=True)
train_fn = theano.function([input_var, target_var], [loss,norm], updates=updates)
test_fn = theano.function([input_var], L.layers.get_output(network, deterministic=True))

#load network
f = open('60cnn.pickle', 'rb')
m = cPickle.load(f)
f.close()
L.layers.set_all_param_values(network, m)

for i in range(0,windows.shape[0]):
    if data[i][0]=='SPY':            
        reshape=np.reshape(windows[i],(1,1,99))
        val_output = test_fn(reshape)
        val_predictions = np.argmax(val_output[0])
        if val_predictions==2:
            val_predictions=-1
        spy_score+=val_predictions
        continue
    if data[i][0]!=previous:
        if count>0:
            underlying_score+=(float(current_total)/float(count))*weight_map[previous]
        previous=data[i][0]
        current_total=0
        count=0
    reshape=np.reshape(windows[i],(1,1,99))
    val_output = test_fn(reshape)
    val_predictions = np.argmax(val_output[0])
    if val_predictions==2:
        val_predictions=-1 #fix negative days
    current_total+=val_predictions
    print data[i][0], data[i][1], data[i][2], 'prediction =', val_predictions
    count+=1

print 'underlying score', underlying_score, 'spy_score', spy_score

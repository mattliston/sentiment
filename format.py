from bs4 import BeautifulSoup
import urllib2
import numpy as np
import re
import csv
import numpy as np
import string
from datetime import timedelta
from datetime import datetime
from six.moves import cPickle

tc={}

with open('WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv') as csvfile: #maps ticker,date to close
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        tc[tuple([row[0],row[1]])]=row[4]

f = open('raw.pickle', 'rb')
data = cPickle.load(f)
f.close()

#data=[] #load article data from csv
#with open('articles.csv') as csvfile:
#    reader=csv.reader(csvfile,delimiter=',')
#    for row in reader:
#        data.append(row)


for i in range(0,len(data)): #fix format of article dates
#    print i
    data[i][1] = str(datetime.strptime(data[i][1],"%B %d, %Y").date())


close=[]
for i in range(0,len(data)):
#    print data[i][1], datetime.strptime(data[i][1],'%Y-%m-%d').date()-timedelta(days=1)
    if (tuple([data[i][0],data[i][1]]) in tc) and (tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=1))]) in tc): 
        close.append(tc[tuple([data[i][0],data[i][1]])]) #normal days
        close.append(tc[tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=1))])])
        
        print data[i][0], data[i][1], data[i][2], tc[tuple([data[i][0],data[i][1]])]       
        print data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=1)) , data[i][2], tc[tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=1))])]
#        print tc[tuple([data[i][0],data[i][1]])], #normal days
#        print tc[tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=1))])]
    
#        print tuple([data[i][0],data[i][1]]) #normal days
#        print tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=1))])

    elif (tuple([data[i][0],data[i][1]]) in tc) and (tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()-timedelta(days=3))]) in tc):
        close.append(tc[tuple([data[i][0],data[i][1]])]) #monday 
        close.append(tc[tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()-timedelta(days=3))])])

    elif (tuple([data[i][0],data[i][1]]) in tc) and (tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=3))]) in tc):
        close.append(tc[tuple([data[i][0],data[i][1]])]) #friday 
        close.append(tc[tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=3))])])

    elif (tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()-timedelta(days=1))]) in tc) and (tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=2))]) in tc):
        close.append(tc[tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()-timedelta(days=1))])]) #saturday 
        close.append(tc[tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=2))])])
    
    elif (tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()-timedelta(days=2))]) in tc) and (tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=1))]) in tc):
        close.append(tc[tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()-timedelta(days=2))])]) #sunday 
        close.append(tc[tuple([data[i][0],str(datetime.strptime(data[i][1],'%Y-%m-%d').date()+timedelta(days=1))])])
    else:
        close.append(1.01) #bad/missing data... adds a positive skew
        close.append(1.0)
#    continue

print len(close)
#exit(0)                
changes=[]
for i in range(0,len(close),2):
    try:
        change = (float(close[i+1])/float(close[i]))-1 #try block to skip bad data
        print change
        changes.append(change)
        print 'changes', changes
    except ValueError:
        print 'value error'
        changes.append(-10) #bad data
print len(changes)
print len(data)
k=0
#quantize
for i in range(0,len(changes)):
    if changes[i]>0:
        changes[i]=1
    elif changes[i]<0:
        changes[i]=-1
    else:
        changes[i]=0
        k+=1

print k
exit(0)
for i in range(0,len(data)):
    print 'data', data[i][2], 'label', changes[i]

#maps each character to a number
character_dict={}
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


d=open('data.pickle','wb')
cPickle.dump(x,d,protocol=cPickle.HIGHEST_PROTOCOL)
d.close()

l=open('label.pickle','wb')
cPickle.dump(changes,l,protocol=cPickle.HIGHEST_PROTOCOL)
l.close()
#np_data=np.asarray(x)
#np_labels=np.asarray(changes)

#myfile = open('labeled_data.csv','wb')
#wr=csv.writer(myfile)
#for i in range(0,len(data)):
#    wr.writerow([data[i][0],data[i][1],data[i][2],x[i],changes[i]])
#    print data[i][0],data[i][1],data[i][2],x[i],changes[i]
    

#only need to make sure each article has correct label, order doesn't matter

#for i in range(0,len(data)):
#    print data[i]

#print len(tickers)
print len(data)


#get all article dates
#get next day changes from wiki closes for those dates #correspond 1 to 1 

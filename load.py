from bs4 import BeautifulSoup
import urllib2
import numpy as np
import re
import csv
import subprocess
import numpy as np
import string
from datetime import timedelta
from datetime import datetime
from six.moves import cPickle

#print datetime.today().date()
#exit(0)

ndays=3
data=[]
last_date={'A':'1999-11-18'}
tickers=[]
prev_sym='A'
prev_date=''
with open('WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv') as csvfile: #loads tickers
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        if (row[0] not in tickers) and (row[0]!='ticker'):
            tickers.append(row[0])
            print row[0]
#            break
        if prev_sym!=row[0]:
            last_date[row[0]]=row[1]
            prev_sym=row[0]
        prev_sym=row[0]
        prev_date=row[1]


for t in tickers: #loads articles
#    for p in range(0,ndays):
    try:
#            url='https://www.benzinga.com/stock-articles/'+t+'/stock?page='+str(p)
        url='https://www.benzinga.com/stock-articles/'+t+'/all'
        print 'url',url
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html)
        a = soup.find_all('div',class_='views-field-title')
#            x = soup.find_all('div',class_='story')
#            print x
#            for b in x:
#                 print b.parent.parent.parent.h3
#                 print b.span.a
#            exit(0)
        for b in a:
#                print b.parent.parent.parent.h3
#                exit(0)
#                print re.findall(r'">(.*?)</a>',str(b.span.a))
#                exit(0)
            date = re.findall(r' (.*?)</h|$',str(b.parent.parent.parent.h3))[0]
            article = re.findall(r'">(.*?)</a>|$',str(b.span.a))[0]
#                print str(datetime.strptime(date,'%B %d, %Y').date()), last_date[t]
#                print date
            if (str(datetime.strptime(date,'%B %d, %Y').date()) < last_date[t]) or (str(datetime.strptime(date,'%B %d, %Y').date())==datetime.today().date()):
                continue
            else:
                data.append([t,date,article]) 
                print t,date,article
    except urllib2.HTTPError:
        continue
print len(data)

f=open('raw.pickle','wb')
cPickle.dump(data,f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()




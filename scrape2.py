from bs4 import BeautifulSoup
import urllib2

tickers = ['SPY']
ndays=3
for t in tickers:
    for p in range(0,ndays):
        url='https://www.benzinga.com/stock-articles/'+t+'/news?page='+str(p)
        print 'url',url
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html)
        #print soup.prettify().encode('ascii','ignore')
        a = soup.find_all('div',class_='views-field-title')
        for b in a:
            print b.parent.parent.parent.h3, b.span.a

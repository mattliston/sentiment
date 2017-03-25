from bs4 import BeautifulSoup
import urllib

tickers = ['SPY']
for i in tickers:
    url = "https://uk.finance.yahoo.com/quote/" + i + "?p="+i
    html = urllib.urlopen(url).read()
    soup = BeautifulSoup(html)
    print soup.prettify()[20000:150000]



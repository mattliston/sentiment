#from bs4 import BeautifulSoup
import re
import urllib

tickers = ['SPY']
for i in tickers:
    url = "https://uk.finance.yahoo.com/quote/" + i + "?p="+i
    html = urllib.urlopen(url).read()
    print type(html)
#    soup = BeautifulSoup(html)
    
    title_data = re.findall(r'title":(.*?),"', html)
    summary_data = re.findall(r'summary":(.*?),"',html)
    print "title data", title_data, "\n"
    print "summary data",summary_data
#    tag = soup["</div><script>"]
#    print tag
#    print soup
    exit(0)
#    for script in soup(["script", "style"]):
#        script.extract()    # rip it out

# get text
    text = soup.get_text()
    print text
    exit(0)
# break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    print(text)
#    text = soup.find_all("script")
#    print data
#    print soup.prettify()[20000:150000]



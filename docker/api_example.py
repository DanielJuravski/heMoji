import requests
from pprint import pprint

app_url = "http://127.0.0.1:5000/"

# get some Hebrew text
news = requests.get(url='http://newsapi.org/v2/top-headlines?country=il&category=entertainment&apiKey=142305a054a54d41adf87c97101f5422').json()
text = news[u'articles'][0][u'title']

# predict emojis for the text above
result = requests.get(url=app_url+text)

pprint(result.json())

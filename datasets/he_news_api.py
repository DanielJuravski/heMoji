import requests

# get some Hebrew text
news = requests.get(url='http://newsapi.org/v2/top-headlines?country=il&category=entertainment&apiKey=142305a054a54d41adf87c97101f5422').json()
important_text = news[u'articles'][0][u'title']

print(important_text)
import requests
import json
r = requests.get('https://newsapi.org/v1/articles?source=cnn&sortBy=top&apiKey=3a47760a55b34dfaa23d897c5d475972')
res=r.json()
results=res['articles']
for result in results:
    dict_news={
        'description':result['description'],
        'title':result['title'],
        'publishedAt':result['publishedAt']
    }
    with open('streaming_news.json', 'a') as f:
        f.write(json.dumps(dict_news))
        f.write('\n')
print(res)
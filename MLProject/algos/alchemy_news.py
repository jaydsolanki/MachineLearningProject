import requests
total_categories = [
                'atheism', 'christian',
                'computer graphics', 'medicine',
                'Microsoft',
                'IBM',
                'mac','Sale','Automobiles',
                'motorcycles',
                'baseball',
                'hockey','politics',
                'guns',
                'mideast','cryptography',
                'electronics',
                'space','religion'
              ]
i=0
while(i<len(total_categories)):
    r = requests.get('https://gateway-a.watsonplatform.net/calls/data/GetNews?outputMode=json&start=now-30d&end=now&count=30&q.enriched.url.concepts.concept.text='+total_categories[i]+'&return=enriched.url.url,enriched.url.title&apikey=c549f5cd7c309fca1cec8e1f725a30a74f34da0f')
    res=r.json()
    print(type(res), res)
    if 'result' in res.keys():
        result=res['result']['docs']
        for data in result:
            with open(total_categories[i]+'.txt', 'a') as f:
                f.write(data['source']['enriched']['url']['title'])
                f.write('\n')
        i+=1
    else:
        import time
        time.sleep(120)
print('Done')

import requests
import json
import sqlite3

conn = sqlite3.connect('../db.sqlite3')
cursor = conn.cursor()

while True:
    try:
        r = requests.get('https://newsapi.org/v1/articles?source=cnn&sortBy=top&apiKey=3a47760a55b34dfaa23d897c5d475972')
        res = r.json()
        results = res['articles']
        for result in results:
            description = result['description']
            title = result['title']
            if type(title)==tuple:
                title = title[0]
            publishedAt = result['publishedAt']
            print("TITLE: "+str(title))
            print("DESCRIPTION: "+str(description))
            print("PUBLISHEDAT: "+str(publishedAt))
            if publishedAt:
                published_date, published_time = publishedAt.split("T")
                published_date = "'"+published_date.replace("'","''")+"'" if published_date else 'null'
                published_date_val = '='+ published_date
                published_time = "'"+published_time.replace("'","''")+"'" if published_time else 'null'
            else:
                published_date_val = ' is null'
                published_date = 'null'
                published_time = 'null'
            description = "'"+description.replace("'","''")+"'" if description else 'null'
            title = "'"+title.replace("'","''")+"'" if title\
                else 'null'
            title_val = '='+title if title!='null' else 'is null'
            cursor.execute("SELECT 1 from live_news where title"+title_val+" and published_date"+published_date_val)
            data = cursor.fetchall()
            if len(data)>0:
                continue
            insert_query = "INSERT INTO live_news (title, published_date, published_time, description) VALUES ("+title+","+published_date+","+published_time+","+description+")"
            cursor.execute(insert_query)
            conn.commit()
            print(insert_query)
            # with open('streaming_news.json', 'a') as f:
            #     f.write(json.dumps(dict_news))
            #     f.write('\n')
    except Exception as e:
        print("Exception encountered: " + str(e))
        cursor.close()
        conn.close()
        break

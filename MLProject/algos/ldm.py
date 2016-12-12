from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from .getdata import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64


n_topics = 20

dataset = get_all_data(categories=total_categories, random_state=1)

def is_float(num):
    try:
        float(num)
        return True
    except:
        return False


def get_top_words(model, feature_names, n_top_words):
    result = []
    for topic_idx, topic in enumerate(model.components_):
        terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        for j in range(len(terms)):
            if is_float(terms[j]):
                terms[j] = "number"+terms[j]
        result.append([topic_idx,terms])
    return result


def topic_modelling(n_features=10000, n_top_words=10):
    global dataset
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,max_features=n_features,stop_words='english')
    tf = tf_vectorizer.fit_transform(dataset.data)
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,learning_method='online',learning_offset=50.,random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    top_words_per_topic = get_top_words(lda, tf_feature_names, n_top_words)
    charts = []
    for i in range(len(top_words_per_topic)):
        charts.append(base64.b64encode(make_image(' '.join(top_words_per_topic[i][1])).getvalue()).decode('ascii'))
    return top_words_per_topic, charts


def make_image(words):
    wordcloud = WordCloud().generate(words)
    plt.imshow(wordcloud)
    plt.axis("off")
    sio = BytesIO()
    plt.savefig(sio, format="png")
    plt.clf()
    plt.close()
    return sio





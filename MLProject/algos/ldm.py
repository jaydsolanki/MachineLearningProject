from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from getdata import *
n_samples = 2000
n_features = 10000
n_topics = 20
n_top_words = 5


dataset = get_all_data(categories=total_categories, random_state=1)
num_docs_per_labels = {}
total_labels = []
total_count = []
for i in dataset.target:
    j = num_docs_per_labels.get(dataset.target_names[i],0)
    j+=1
    num_docs_per_labels[dataset.target_names[i]] = j

for key in num_docs_per_labels.keys():
    total_labels.append(key)
    total_count.append(num_docs_per_labels[key])


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

# Use tf-idf features for NMF.
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=1,max_features=n_features,stop_words='english')
tfidf_vectorizer = CountVectorizer(analyzer='word', stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(dataset.data)

# Use tf (raw term count) features for LDA.
# tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1,max_features=n_features,stop_words='english')
tf_vectorizer = CountVectorizer(analyzer='word', stop_words='english')
tf = tf_vectorizer.fit_transform(dataset.data)

# Fit the NMF model
# nmf = NMF(n_components=n_topics, random_state=1,alpha=.1, l1_ratio=.5).fit(tfidf)
#
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,learning_method='online',learning_offset=50.,random_state=0)
results = lda.fit(tf)
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
pass

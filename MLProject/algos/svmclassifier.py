# import numpy as np
# X=['send us your password','send us your review','review your password','review us' ,'send your password','send us your account']
# y = ['spam', 'ham', 'ham', 'spam', 'spam', 'spam']
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# clf.fit(np.array(X).reshape(-1,1), y)
#
# print(clf.predict('your review'))

from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med',
'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware',
'comp.windows.x','misc.forsale','rec.autos',
'rec.motorcycles',
'rec.sport.baseball',
'rec.sport.hockey','talk.politics.misc',
'talk.politics.guns',
'talk.politics.mideast','sci.crypt',
'sci.electronics',
'sci.space','talk.religion.misc']
twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer='word', ngram_range=(1, 3),stop_words='english')
X_train_counts = count_vect.fit_transform(twenty_train.data)
#X_train_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#X_train_tf.shape
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge",  alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf, twenty_train.target)

docs_new = ['New drug to cure hairfall', 'OpenGL on the GPU is fast','Microsoft releases anniversary updates for the latest version of windows'
            ,'Apple introduces new touchpad for macbook','Tsunami hits Japan']
X_new_counts = count_vect.transform(docs_new)
X_new_tf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

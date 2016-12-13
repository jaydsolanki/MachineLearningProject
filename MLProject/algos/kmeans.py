# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

from optparse import OptionParser
import sys
from time import time
import numpy as np
from .getdata import *


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


def cluster(num_features=10000, random_state=3):
    # parse commandline arguments
    op = OptionParser()
    global total_labels
    global total_count
    op.add_option("--lsa",
                  dest="n_components", type="int",
                  help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch",
                  action="store_false", dest="minibatch", default=True,
                  help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=num_features,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports inside k-means algorithm.")

    (opts, args) = op.parse_args()
    # if len(args) > 0:
    #     op.error("this script takes no arguments.")
    #     sys.exit(1)

    global dataset
    categories = None
    labels = dataset.target
    true_k = np.unique(labels).shape[0]

    print("Extracting features from the training dataset using a sparse vectorizer")
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english', non_negative=True,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words='english',
                                           non_negative=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=opts.use_idf)
    X = vectorizer.fit_transform(dataset.data)

    if opts.n_components:
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)
        explained_variance = svd.explained_variance_ratio_.sum()

    # Do the actual clustering
    if opts.minibatch:
        km = MiniBatchKMeans(max_iter = 1000, n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=opts.verbose, random_state=random_state)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=10000, n_init=1, verbose=opts.verbose, random_state=random_state)

    results = km.fit(X)
    homogeneity_score = metrics.homogeneity_score(labels, km.labels_)
    completeness_score = metrics.completeness_score(labels, km.labels_)
    v_measure_score = metrics.v_measure_score(labels, km.labels_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    if not opts.use_hashing:
        if opts.n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
    return homogeneity_score, completeness_score, v_measure_score, results.counts_.tolist(), total_labels, total_count

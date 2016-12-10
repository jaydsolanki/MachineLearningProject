from sklearn.datasets import fetch_20newsgroups
total_categories = ['alt.atheism',
                     'comp.graphics',
                     'comp.os.ms-windows.misc',
                     'comp.sys.ibm.pc.hardware',
                     'comp.sys.mac.hardware',
                     'comp.windows.x',
                     'misc.forsale',
                     'rec.autos',
                     'rec.motorcycles',
                     'rec.sport.baseball',
                     'rec.sport.hockey',
                     'sci.crypt',
                     'sci.electronics',
                     'sci.med',
                     'sci.space',
                     'soc.religion.christian',
                     'talk.politics.guns',
                     'talk.politics.mideast',
                     'talk.politics.misc',
                     'talk.religion.misc']

total_categories_display = []

for c in total_categories:
    total_categories_display.append(c.replace("."," "))


def get_train_data(categories=total_categories, random_state=1):
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=random_state,remove=('headers', 'footers', 'quotes'))
    return twenty_train


def get_test_data(categories=total_categories, random_state=1):
    twenty_train = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=random_state,remove=('headers', 'footers', 'quotes'))
    return twenty_train


def get_all_data(categories=total_categories, random_state=1):
    twenty_train = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=random_state,remove=('headers', 'footers', 'quotes'))
    return twenty_train

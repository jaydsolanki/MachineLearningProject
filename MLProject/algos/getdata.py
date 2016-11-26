from sklearn.datasets import fetch_20newsgroups
total_categories = [
                'alt.atheism', 'soc.religion.christian',
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
                'sci.space','talk.religion.misc'
              ]


def get_train_data(categories=total_categories, random_state=1):
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=random_state)
    return twenty_train


def get_test_data(categories=total_categories, random_state=1):
    twenty_train = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=random_state)
    return twenty_train


def get_all_data(categories=total_categories, random_state=1):
    twenty_train = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=random_state)
    return twenty_train

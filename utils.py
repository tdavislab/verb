from sklearn.decomposition import PCA
from vectors import compute_weat_score


def process_seedwords(seedword_string):
    return [seedword.strip() for seedword in seedword_string.split(',')]


def make_projector(method='PCA'):
    if method == 'PCA':
        projector = PCA(n_components=2)
    else:
        raise AttributeError('Unknown method type for dimensionality reduction')
    return projector


def get_weat_score(embedding, seedwords1, seedwords2):
    male_words = {'man', 'male', 'boy', 'brother', 'him', 'his', 'son'}
    female_words = {'woman', 'female', 'girl', 'brother', 'her', 'hers', 'daughter'}
    return compute_weat_score(embedding, male_words, female_words, seedwords1, seedwords2)


def project_to_2d(projector, embedding, wordlist):
    return projector.transform(embedding.get_many(wordlist)).tolist()

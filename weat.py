import numpy as np


def unit_vector(vec):
    """
    Returns unit vector
    """
    return vec / np.linalg.norm(vec)


def cos_sim(v1, v2):
    """
    Returns cosine of the angle between two vectors
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)


def weat_association(W, A, B):
    """
    Returns association of the word w in W with the attribute for WEAT score.
    s(w, A, B)
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: (len(W), ) shaped numpy ndarray. each rows represent association of the word w in W
    """
    return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)


def weat_score(X, Y, A, B):
    """
    Returns WEAT score
    X, Y, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: this function assumes that there's no intersection word between X and Y
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEAT score
    """

    x_association = weat_association(X, A, B)
    y_association = weat_association(Y, A, B)

    tmp1 = np.mean(x_association, axis=-1) - np.mean(y_association, axis=-1)
    tmp2 = np.std(np.concatenate((x_association, y_association), axis=0))
    return tmp1 / tmp2

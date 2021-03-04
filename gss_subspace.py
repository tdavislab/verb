import numpy as np
import math
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import spearmanr

gr = (math.sqrt(5) + 1) / 2  # golden ratiodef projection(u,v): #u is any vector in the embedding, v is the bias direction


def projection(u, v):  # u is any vector in the embedding, v is the bias direction
    u1 = u - np.dot(u, v) * v  # - np.dot(u,v[1])*v[1]
    return u1


# load vectors
# load the corresponding wordlist into an array


# loading words for WEAT (gender v/s occupations)
# WX = ['male', 'man', 'boy', 'brother', 'him', 'his', 'son']
# WY = ['female', 'woman', 'girl', 'sister', 'her', 'hers', 'daughter']
# WA = ['doctor', 'engineer', 'lawyer', 'mathematician', 'banker']
# WB = ['receptionist', 'homemaker', 'nurse', 'dancer', 'maid']


def cosine1(x, y):
    return np.dot(x, y.T)


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
    # tmp2 = np.std(x_association)

    return tmp1 / tmp2


# Functions to feed into subspace determining function start with f_

def f_weat(embedding, x1, vec):
    WX = ['male', 'man', 'boy', 'brother', 'him', 'his', 'son']
    WY = ['female', 'woman', 'girl', 'sister', 'her', 'hers', 'daughter']
    WA = ['doctor', 'engineer', 'lawyer', 'mathematician', 'banker']
    WB = ['receptionist', 'homemaker', 'nurse', 'dancer', 'maid']

    dim = embedding.get(WA[0]).vector.shape[0]

    vector = x1 - vec
    if np.linalg.norm(vector) != 0.0:
        vector = vector / np.linalg.norm(vector)
    VA = np.zeros((len(WA), dim))
    VB = np.zeros((len(WB), dim))
    VX = np.zeros((len(WX), dim))
    VY = np.zeros((len(WY), dim))
    for i in range(len(WA)):
        VA[i] = projection(embedding.get(WA[i]).vector, vector)
    for i in range(len(WB)):
        VB[i] = projection(embedding.get(WB[i]).vector, vector)
    for i in range(len(WX)):
        VX[i] = projection(embedding.get(WX[i]).vector, vector)
    for i in range(len(WY)):
        VY[i] = projection(embedding.get(WY[i]).vector, vector)
    return weat_score(VX, VY, VA, VB)



# Subspace determination given
# f: funtion to optimize over
# x1: the fixed end of the direction (e.g., the male average
# a,b: 2 points whose weighted average we find to optimize f
# tol: when f1 does not change by more than tol, we break out of loop
def gss(f, a, b, x1, embedding, tol=1e-5):
    """Golden section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]
    """

    c = b - np.true_divide((b - a), gr)
    d = a + np.true_divide((b - a), gr)
    while np.linalg.norm(c - d) > tol:
        if f(embedding, x1, c) < f(embedding, x1, d):
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - np.true_divide((b - a), gr)
        d = a + np.true_divide((b - a), gr)

    return (b + a) / 2


# Define x1 and y1 as initial points/means

# x1 = vec['he']
# y1 = vec['she']  # put in initial vectors here

"""
#Calling the function over a list of points added in Y
v = np.zeros((2,300))
for k in range(2):   
	Y = np.random.permutation(Y)
	a = A[wl.index(Y[0])]
	b = A[wl.index(Y[1])]
	for i in range(2,len(Y)):
		#print(i-2, Y[i-2], Y[i-1], f(gss(f,a,b,tol=1e-10)))
		a = gss(f_weat,a,b, x, tol=1e-5)
		b = A[wl.index(Y[i])]
		print(i)
	v[k] = a
	print(k, f(x,a))
print('avg',f(x,np.mean(v,0)))
l = np.mean(v,0)


#Calling the function over a list of points added in X
v = np.zeros((2,300))
for k in range(2):   
	X = np.random.permutation(X)
	a = A[wl.index(X[0])]
	b = A[wl.index(X[1])]
	for i in range(2,len(X)):
		#print(i-2, Y[i-2], Y[i-1], f(gss(f,a,b,tol=1e-10)))
		a = gss(f_weat,a,b, l, tol=1e-5)
		b = A[wl.index(X[i])]
		print(i)
	v[k] = a
	print(k, f(l,a))
print('avg',f(l,np.mean(v,0)))
"""

# # Calling the function for 1 point added in X
# a = x1
# b =  # add point
# # y1 is predetermined and not changed
# x1 = gss(f_weat, a, b, tol=1e-5)
# direction = x1 - y1;
# direction = direction / np.linalg.norm(direction)
#
# # Calling the function for 1 point added in Y
# a = y1
# b =  # add point
# # x1 is predetermined and not changed
# x1 = gss(f_weat, a, b, tol=1e-5)
# direction = x1 - y1;
# direction = direction / np.linalg.norm(direction)

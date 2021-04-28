import pickle

import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tqdm import tqdm

from dynamicProj import generateFullDynamicProjPath
from gss_subspace import f_weat, gss
from weat import weat_score


class WordVector:
    def __init__(self, index, word, vector):
        self.index = index
        self.word = word
        self.vector = vector

    def __str__(self):
        return f'Group={self.group}, Word="{self.word}"'

    def __repr__(self):
        return f'WordVector({self.word}, {self.vector}, {self.group})'


# Embedding class
# ----------------------------------------------------
class Embedding:
    def __init__(self, path, metadata=None, group=None, limit=20000):
        # either create an empty embedding (useful for creating debiased embeddings)
        if path is None:
            self.word_vectors = []
        # or read embeddings from disk
        else:
            self.word_vectors = read_embeddings(path, limit=limit)
            # self.word_vectors = read_merchant_embedding(path, metadata, group, limit=limit)

    def get(self, word, color=''):
        """
        Get WordVector object for single word
        """
        return self.word_vectors[word]

    def get_many(self, words):
        """
        Get list of WordVector objects for a list of words
        """
        return [self.word_vectors[word] for word in words]

    def get_vecs(self, words):
        """
        Get numpy array of vectors for a given list of words
        """
        return np.vstack([self.word_vectors[word].vector for word in words])

    def vectors(self):
        return np.vstack([wv.vector for wv in self.word_vectors.values()])

    def words(self):
        return [wv.word for wv in self.word_vectors.values()]

    def update_vectors(self, words, new_vectors):
        for i, word in enumerate(words):
            self.word_vectors[word].vector = new_vectors[i]

    def normalize(self, center=False):
        vectors = self.vectors()
        if center:
            mean = np.mean(vectors, axis=0)
            normed_vectors = (vectors - mean) / np.linalg.norm(vectors - mean)
        else:
            normed_vectors = vectors / np.linalg.norm(vectors)
        self.update_vectors(self.words(), normed_vectors)

    def merge(self, other_embedding):
        self.word_vectors = {**self.word_vectors, **other_embedding.word_vectors}

    def to_csv_list(self):
        return [[word] + self.word_vectors[word].vector.tolist() for word in self.word_vectors]


# Alignment
def closed_form_alignment(A, B):
    # centering
    Aprime = A - np.mean(A, 0)
    Bprime = B - np.mean(B, 0)

    # rotation
    N = np.zeros((np.shape(A)[1], np.shape(A)[1]));
    for i in range(A.shape[0]):
        N = N + np.outer(Bprime[i], Aprime[i])

    U, S, V = np.linalg.svd(N)
    R = np.matmul(U, V)
    newBprime = np.matmul(Bprime, R)

    # scaling
    s1 = 0.0
    s2 = 0.0

    for i in range(0, A.shape[0]):
        s1 = s1 + np.dot(Aprime[i], newBprime[i])
        s2 = s2 + np.dot(Bprime[i], Bprime[i])

    s = s1 / s2

    # output matrix : B oriented onto A
    newB = s * newBprime + np.mean(A, 0)

    return newB


class AlignmentProjector:
    def __init__(self, emb1_obj: Embedding, emb2_obj: Embedding):
        self.emb1 = emb1_obj
        self.emb2 = emb2_obj
        self.vecs1 = None
        self.vecs2 = None
        self.wordlist = None

    def compute_2d(self, wordlist, align=True):
        def get_aligned_vecs(vecs, corresponding_words, target_words):
            # get the new aligned vectors for wordlist
            aligned_vecs = [vecs[corresponding_words.index(target_word)] for target_word in target_words]
            return np.array(aligned_vecs)

        # Fit PCA to words
        # Multiple strategies are possible
        # 1. PCA on space of emb1 or emb2
        # 2. PCA on space of emb1 + emb2 (vstack vector)

        # wordlist = list(set(self.emb1.words()).intersection(set(self.emb2.words())))[:50]
        # wordlist = self.emb1.words()[:50]
        common_words = list(set(self.emb1.words()).intersection(set(self.emb2.words())))

        # emb1_vecs = self.emb1.get_vecs(wordlist)
        # emb2_vecs = self.emb2.get_vecs(wordlist)
        emb2_vecs = self.emb2.get_vecs(common_words)
        emb1_vecs = self.emb1.get_vecs(common_words)

        # Perform closed-form alignment
        if align:
            emb2_vecs = closed_form_alignment(emb1_vecs, emb2_vecs)

        # Go with PCA on space of emb1 for now
        projector = PCA(n_components=2)
        projector.fit(np.vstack([get_aligned_vecs(emb1_vecs, common_words, wordlist), get_aligned_vecs(emb2_vecs, common_words, wordlist)]))
        # projector.fit(emb1_vecs)

        self.vecs1 = projector.transform(get_aligned_vecs(emb1_vecs, common_words, wordlist))
        self.vecs2 = projector.transform(get_aligned_vecs(emb2_vecs, common_words, wordlist))
        self.wordlist = wordlist

        return self

    def convert_to_payload(self):
        payload = {'emb1': [], 'emb2': []}

        for i, word in enumerate(self.wordlist):
            payload['emb1'].append({'label': word, 'x': self.vecs1[i][0], 'y': self.vecs1[i][1], 'group': 1})
            payload['emb2'].append({'label': word, 'x': self.vecs2[i][0], 'y': self.vecs2[i][1], 'group': 2})

        return payload


# Debiaser base class
# ----------------------------------------------------
class Debiaser:
    def __init__(self, base_embedding: Embedding, debiased_embedding: Embedding, app_instance):
        self.base_emb = base_embedding
        self.debiased_emb = debiased_embedding
        self.animator = Animator()
        self.app_instance = app_instance

    def debias(self, bias_direction, seedwords1, seedwords2, evalwords):
        # Compute debiased embedding and create animation steps here
        raise NotImplementedError('This method should not be called from Debiaser object.')


class LinearDebiaser(Debiaser):
    def debias(self, bias_direction, seedwords1, seedwords2, evalwords):
        # Debias the embedding
        debiased_vectors = self.base_emb.vectors() - self.base_emb.vectors().dot(bias_direction.reshape(-1, 1)) * bias_direction
        self.debiased_emb.update_vectors(self.base_emb.words(), debiased_vectors)

        # ---------------------------------------------------------
        # Step 0 - PCA of points in the original word vector space
        # ---------------------------------------------------------
        prebase_projector = self.animator.add_projector(PCA(n_components=2), name='prebase_projector')
        prebase_projector.fit(self.base_emb, seedwords1 + seedwords2)

        step0 = self.animator.add_anim_step()
        step0.add_points(prebase_projector.project(self.base_emb, seedwords1, group=1))
        step0.add_points(prebase_projector.project(self.base_emb, seedwords2, group=2))
        step0.add_points(prebase_projector.project(self.base_emb, evalwords, group=3))
        step0.add_points(prebase_projector.project(self.base_emb, [], group=0, direction=bias_direction))

        # ---------------------------------------------------------
        # Step 1 - Project points such that bias direction is aligned with the x-axis
        # ---------------------------------------------------------
        base_projector = self.animator.add_projector(BiasPCA(), name='base_projector')
        base_projector.fit(self.base_emb, seedwords1 + seedwords2 + evalwords, bias_direction=bias_direction)

        step1 = self.animator.add_anim_step(camera_step=True)
        step1.add_points(base_projector.project(self.base_emb, seedwords1, group=1))
        step1.add_points(base_projector.project(self.base_emb, seedwords2, group=2))
        step1.add_points(base_projector.project(self.base_emb, evalwords, group=3))
        step1.add_points(base_projector.project(self.base_emb, [], group=0, direction=bias_direction))

        # ---------------------------------------------------------
        # Step 2 - Show the bias-x aligned projection of the debiased embedding
        # ---------------------------------------------------------

        step2 = self.animator.add_anim_step()
        step2.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
        step2.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
        step2.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))
        step2.add_points(base_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))

        # ---------------------------------------------------------
        # Step 3 - Project to the space of debiased embeddings
        # ---------------------------------------------------------
        debiased_projector = self.animator.add_projector(PCA(n_components=2), name='debiased_projector')
        debiased_projector.fit(self.debiased_emb, seedwords1 + seedwords2)

        step3 = self.animator.add_anim_step(camera_step=True)
        step3.add_points(debiased_projector.project(self.debiased_emb, seedwords1, group=1))
        step3.add_points(debiased_projector.project(self.debiased_emb, seedwords2, group=2))
        step3.add_points(debiased_projector.project(self.debiased_emb, evalwords, group=3))
        step3.add_points(debiased_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))

        self.animator.make_transition(step0, step1)
        self.animator.make_transition(step1, step2)
        self.animator.make_transition(step2, step3)


class HardDebiaser(Debiaser):
    def debias(self, bias_direction, seedwords1, seedwords2, evalwords, equalize_set=None):
        equalize_words = [list(x) for x in list(zip(*equalize_set))]

        # ---------------------------------------------------------
        # Step 0 - PCA of points in the original word vector space
        # ---------------------------------------------------------
        prebase_projector = self.animator.add_projector(PCA(n_components=2), name='prebase_projector')
        prebase_projector.fit(self.base_emb, seedwords1 + seedwords2 + evalwords + equalize_words[0] + equalize_words[1])

        step0 = self.animator.add_anim_step()
        step0.add_points(prebase_projector.project(self.base_emb, seedwords1, group=1))
        step0.add_points(prebase_projector.project(self.base_emb, seedwords2, group=2))
        step0.add_points(prebase_projector.project(self.base_emb, evalwords, group=3))
        step0.add_points(prebase_projector.project(self.base_emb, equalize_words[0], group=4))
        step0.add_points(prebase_projector.project(self.base_emb, equalize_words[1], group=5))
        step0.add_points(prebase_projector.project(self.base_emb, [], group=0, direction=bias_direction))

        # ---------------------------------------------------------
        # Step 1 - Project points such that bias direction is aligned with the x-axis
        # ---------------------------------------------------------
        base_projector = self.animator.add_projector(BiasPCA(), name='base_projector')
        base_projector.fit(self.base_emb, seedwords1 + seedwords2 + evalwords + equalize_words[0] + equalize_words[1], bias_direction=bias_direction)

        step1 = self.animator.add_anim_step(camera_step=True)
        step1.add_points(base_projector.project(self.base_emb, seedwords1, group=1))
        step1.add_points(base_projector.project(self.base_emb, seedwords2, group=2))
        step1.add_points(base_projector.project(self.base_emb, evalwords, group=3))
        step1.add_points(base_projector.project(self.base_emb, equalize_words[0], group=4))
        step1.add_points(base_projector.project(self.base_emb, equalize_words[1], group=5))
        step1.add_points(base_projector.project(self.base_emb, [], group=0, direction=bias_direction))

        # ---------------------------------------------------------
        # Step 2 - Remove gender component from the evalwords except gender-specific words
        # ---------------------------------------------------------

        # for i, word in enumerate(set(evalwords + equalize_words[0] + equalize_words[1])):
        #     # remove bias direction from dataset
        #     self.debiased_emb.word_vectors[word].vector = remove_component(self.base_emb.word_vectors[word].vector, bias_direction)

        for i, word in enumerate(set(self.base_emb.words())):
            # remove bias direction from dataset
            if word not in seedwords1 + seedwords2:
                self.debiased_emb.word_vectors[word].vector = remove_component(self.base_emb.word_vectors[word].vector, bias_direction)

        step2 = self.animator.add_anim_step()
        step2.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
        step2.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
        step2.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))
        step2.add_points(base_projector.project(self.debiased_emb, equalize_words[0], group=4))
        step2.add_points(base_projector.project(self.debiased_emb, equalize_words[1], group=5))
        step2.add_points(base_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))

        # Equalize words in equalize_set such that they are equidistant to set defining the gender direction
        for a, b in equalize_set:
            y = remove_component((self.base_emb.word_vectors[a].vector + self.base_emb.word_vectors[b].vector) / 2, bias_direction)
            z = np.sqrt(1 - np.linalg.norm(y) ** 2)

            if (self.base_emb.word_vectors[a].vector - self.base_emb.word_vectors[b].vector).dot(bias_direction) < 0:
                z = -z

            self.debiased_emb.word_vectors[a].vector = z * bias_direction + y
            self.debiased_emb.word_vectors[b].vector = -z * bias_direction + y

        # self.debiased_emb.normalize()

        # ---------------------------------------------------------
        # Step 3 - Compute new projection of the words in equalize set so that they are equidistant to both clusters
        # ---------------------------------------------------------
        step3 = self.animator.add_anim_step()
        step3.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
        step3.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
        step3.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))
        step3.add_points(base_projector.project(self.debiased_emb, equalize_words[0], group=4))
        step3.add_points(base_projector.project(self.debiased_emb, equalize_words[1], group=5))
        step3.add_points(base_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))

        # Step 4 - Reorient the embeddings back to the debiased space
        # ---------------------------------------------------------
        debiased_projector = self.animator.add_projector(PCA(n_components=2), name='debiased_projector')
        # debiased_projector.fit(self.debiased_emb, seedwords1 + seedwords2 + evalwords + equalize_words[0] + equalize_words[1])
        debiased_projector.fit(self.debiased_emb, seedwords1 + seedwords2)

        step4 = self.animator.add_anim_step(camera_step=True)
        step4.add_points(debiased_projector.project(self.debiased_emb, seedwords1, group=1))
        step4.add_points(debiased_projector.project(self.debiased_emb, seedwords2, group=2))
        step4.add_points(debiased_projector.project(self.debiased_emb, evalwords, group=3))
        step4.add_points(debiased_projector.project(self.debiased_emb, equalize_words[0], group=4))
        step4.add_points(debiased_projector.project(self.debiased_emb, equalize_words[1], group=5))
        step4.add_points(debiased_projector.project(self.debiased_emb, [], group=0, direction=bias_direction - bias_direction + 1e-8))


class OscarDebiaser(Debiaser):
    def debias(self, bias_direction, seedwords1, seedwords2, evalwords, orth_subspace_words, use2d=True, bias_method=None):
        if use2d:
            # 1. compute the primary subspace direction and the secondary subspace direction in original space: v1, v2
            # 2. v2' = orth_component(v2, v1) and make unit vector
            # 3. v1 and v2' define the new x and y-axis respectively
            # 4. project n-dimensional points to v1 and v2'
            v1 = get_bias_direction(self.base_emb, seedwords1, seedwords2, bias_method)
            v2 = get_bias_direction(self.base_emb, orth_subspace_words, None, 'PCA')
            theta = angle_between(v1, v2)
            if theta > np.pi / 2:
                v2 = -v2
            v2_prime = v2 - v1 * (v2.dot(v1))
            v2_prime = v2_prime / np.linalg.norm(v2_prime)

            # ---------------------------------------------------------
            # Step 0 - PCA of points in the original word vector space
            # ---------------------------------------------------------
            prebase_projector = self.animator.add_projector(PCA(n_components=2), name='prebase_projector')
            prebase_projector.fit(self.base_emb, seedwords1 + seedwords2 + orth_subspace_words)

            step0 = self.animator.add_anim_step()
            step0.add_points(prebase_projector.project(self.base_emb, seedwords1, group=1))
            step0.add_points(prebase_projector.project(self.base_emb, seedwords2, group=2))
            step0.add_points(prebase_projector.project(self.base_emb, evalwords, group=3))
            step0.add_points(prebase_projector.project(self.base_emb, orth_subspace_words, group=4))
            step0.add_points(prebase_projector.project(self.base_emb, [], group=0, direction=v1))
            step0.add_points(prebase_projector.project(self.base_emb, [], group=0, direction=v2, concept_idx=2))

            vecs = self.base_emb.vectors()
            x_coord = vecs.dot(v1)
            y_coord = vecs.dot(v2_prime)
            self.base_emb.update_vectors(self.base_emb.words(), np.vstack([x_coord, y_coord]).T)
            self.debiased_emb.update_vectors(self.base_emb.words(), self.base_emb.vectors())

            bias_direction = np.array([v1.dot(v1), v1.dot(v2_prime)])
            bias_direction = bias_direction / np.linalg.norm(bias_direction)
            orth_direction = np.array([v2.dot(v1), v2.dot(v2_prime)])
            orth_direction = orth_direction / np.linalg.norm(orth_direction)
            orth_direction_prime = np.array([v2_prime.dot(v1), v2_prime.dot(v2_prime)])
            orth_direction_prime = orth_direction_prime / np.linalg.norm(orth_direction_prime)

            # ---------------------------------------------------------
            # Step 1 - Project points such that bias direction is aligned with the x-axis
            # ----------------------------------------------------------
            base_projector = self.animator.add_projector(CoordinateProjector(), name='base_projector')
            base_projector.fit(self.base_emb, seedwords1 + seedwords2 + orth_subspace_words, bias_direction=bias_direction,
                               secondary_direction=orth_direction_prime)

            step1 = self.animator.add_anim_step(camera_step=True)
            step1.add_points(base_projector.project(self.base_emb, seedwords1, group=1))
            step1.add_points(base_projector.project(self.base_emb, seedwords2, group=2))
            step1.add_points(base_projector.project(self.base_emb, evalwords, group=3))
            step1.add_points(base_projector.project(self.base_emb, orth_subspace_words, group=4))
            step1.add_points(base_projector.project(self.base_emb, [], group=0, direction=bias_direction))
            step1.add_points(base_projector.project(self.base_emb, [], group=0, direction=orth_direction, concept_idx=2))

            rot_matrix = self.gs_constrained2d_new(np.identity(bias_direction.shape[0]), bias_direction, orth_direction)

            for word in set(seedwords1 + seedwords2 + evalwords + orth_subspace_words):
                self.debiased_emb.word_vectors[word].vector = self.correction2d_new(rot_matrix, bias_direction, orth_direction,
                                                                                    self.base_emb.word_vectors[word].vector)

            # for word in self.base_emb.words():
            #     self.debiased_emb.word_vectors[word].vector = self.correction2d_new(rot_matrix, bias_direction, orth_direction,
            #                                                                         self.base_emb.word_vectors[word].vector)

            base_projector = self.animator.add_projector(CoordinateProjector(), name='base_projector')
            base_projector.fit(self.base_emb, seedwords1 + seedwords2 + orth_subspace_words, bias_direction=bias_direction,
                               secondary_direction=orth_direction_prime)

            step2 = self.animator.add_anim_step()
            step2.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
            step2.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
            step2.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))
            step2.add_points(base_projector.project(self.debiased_emb, orth_subspace_words, group=4))
            step2.add_points(base_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))
            step2.add_points(base_projector.project(self.debiased_emb, [], group=0, direction=orth_direction_prime, concept_idx=2))

            # ---------------------------------------------------------
            # Step 3 - Project points such the orth direction is aligned with y-axis
            # ---------------------------------------------------------
            self.animator.add_projector(PCA(n_components=2), name='debiased_projector')
            debiased_projector = self.animator.projectors['debiased_projector']
            debiased_projector.fit(self.debiased_emb, seedwords1 + seedwords2 + orth_subspace_words)

            step3 = self.animator.add_anim_step(camera_step=True)
            step3.add_points(debiased_projector.project(self.debiased_emb, seedwords1, group=1))
            step3.add_points(debiased_projector.project(self.debiased_emb, seedwords2, group=2))
            step3.add_points(debiased_projector.project(self.debiased_emb, evalwords, group=3))
            step3.add_points(debiased_projector.project(self.debiased_emb, orth_subspace_words, group=4))
            step3.add_points(debiased_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))
            step3.add_points(debiased_projector.project(self.debiased_emb, [], group=0, direction=orth_direction_prime, concept_idx=2))

            # Now reset embedding to n-dimensional version
            self.app_instance.reload_embeddings()

            bias_direction = get_bias_direction(self.app_instance.base_embedding, seedwords1, seedwords2, bias_method)
            orth_direction = get_bias_direction(self.app_instance.base_embedding, orth_subspace_words, None, 'PCA')
            rot_matrix = self.gs_constrained(np.identity(bias_direction.shape[0]), bias_direction, orth_direction)

            for word in self.app_instance.base_embedding.words():
                self.app_instance.debiased_embedding.word_vectors[word].vector = self.correction(rot_matrix, bias_direction, orth_direction,
                                                                                                 self.app_instance.base_embedding.word_vectors[
                                                                                                     word].vector)
        else:
            orth_direction = bias_pca(self.base_emb, orth_subspace_words)

            # ---------------------------------------------------------
            # Step 0 - PCA of points in the original word vector space
            # ---------------------------------------------------------
            prebase_projector = self.animator.add_projector(PCA(n_components=2), name='prebase_projector')
            prebase_projector.fit(self.base_emb, seedwords1 + seedwords2)

            step0 = self.animator.add_anim_step()
            step0.add_points(prebase_projector.project(self.base_emb, seedwords1, group=1))
            step0.add_points(prebase_projector.project(self.base_emb, seedwords2, group=2))
            step0.add_points(prebase_projector.project(self.base_emb, evalwords, group=3))
            step0.add_points(prebase_projector.project(self.base_emb, orth_subspace_words, group=4))
            step0.add_points(prebase_projector.project(self.base_emb, [], group=0, direction=bias_direction))
            step0.add_points(prebase_projector.project(self.base_emb, [], group=0, direction=orth_direction, concept_idx=2))

            # ---------------------------------------------------------
            # Step 1 - Project points such that bias direction is aligned with the x-axis
            # ----------------------------------------------------------
            base_projector = self.animator.add_projector(BiasPCA(), name='base_projector')
            base_projector.fit(self.base_emb, seedwords1 + seedwords2, bias_direction=bias_direction, secondary_direction=orth_direction)

            step1 = self.animator.add_anim_step(camera_step=True)
            step1.add_points(base_projector.project(self.base_emb, seedwords1, group=1))
            step1.add_points(base_projector.project(self.base_emb, seedwords2, group=2))
            step1.add_points(base_projector.project(self.base_emb, evalwords, group=3))
            step1.add_points(base_projector.project(self.base_emb, orth_subspace_words, group=4))
            step1.add_points(base_projector.project(self.base_emb, [], group=0, direction=bias_direction))
            step1.add_points(base_projector.project(self.base_emb, [], group=0, direction=orth_direction, concept_idx=2))

            # ---------------------------------------------------------
            # Step 2 - Make orth_direction orthogonal to bias direction
            # ---------------------------------------------------------
            rot_matrix = self.gs_constrained(np.identity(bias_direction.shape[0]), bias_direction, orth_direction)

            for word in seedwords1 + seedwords2 + evalwords + orth_subspace_words:
                self.debiased_emb.word_vectors[word].vector = self.correction(rot_matrix, bias_direction, orth_direction,
                                                                              self.base_emb.word_vectors[word].vector)

            orth_direction_prime = basis(np.vstack([bias_direction, orth_direction]))

            step2 = self.animator.add_anim_step()
            step2.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
            step2.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
            step2.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))
            step2.add_points(base_projector.project(self.debiased_emb, orth_subspace_words, group=4))
            step2.add_points(base_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))
            step2.add_points(base_projector.project(self.debiased_emb, [], group=0, direction=orth_direction_prime, concept_idx=2))

            # ---------------------------------------------------------
            # Step 3 - Project points such the orth direction is aligned with y-axis
            # ---------------------------------------------------------
            self.animator.add_projector(PCA(n_components=2), name='debiased_projector')
            debiased_projector = self.animator.projectors['debiased_projector']
            debiased_projector.fit(self.debiased_emb, seedwords1 + seedwords2 + orth_subspace_words)

            step3 = self.animator.add_anim_step(camera_step=True)
            step3.add_points(debiased_projector.project(self.debiased_emb, seedwords1, group=1))
            step3.add_points(debiased_projector.project(self.debiased_emb, seedwords2, group=2))
            step3.add_points(debiased_projector.project(self.debiased_emb, evalwords, group=3))
            step3.add_points(debiased_projector.project(self.debiased_emb, orth_subspace_words, group=4))
            step3.add_points(debiased_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))
            step3.add_points(debiased_projector.project(self.debiased_emb, [], group=0, direction=orth_direction_prime, concept_idx=2))

    @staticmethod
    def correction(rotation_matrix, v1, v2, x):
        def rotation(dir1, dir2, input_vec):
            dir1 = np.asarray(dir1).reshape(-1)
            dir2 = np.asarray(dir2).reshape(-1)
            input_vec = np.asarray(input_vec).reshape(-1)
            v2_p = basis(np.vstack((dir1, dir2)))
            x_p = input_vec[2:len(input_vec)]
            input_vec = (np.dot(input_vec, dir1), np.dot(input_vec, v2_p))
            dir2 = (np.matmul(dir2, dir1.T), np.sqrt(1 - (np.matmul(dir2, dir1.T) ** 2)))
            dir1 = (1, 0)
            theta_x = 0.0

            # theta = np.abs(np.arccos(np.dot(dir1, dir2)))
            # theta_p = (np.pi / 2.0) - theta
            theta_p = np.arccos(np.dot(v1, v2))
            theta = np.abs(theta_p - np.pi / 2)

            phi = np.arccos(np.dot(dir1, input_vec / np.linalg.norm(input_vec)))
            d = np.dot([0, 1], input_vec / np.linalg.norm(input_vec))
            if phi < theta_p and d > 0:
                theta_x = theta * (phi / theta_p)
            elif phi > theta_p and d > 0:
                theta_x = theta * ((np.pi - phi) / (np.pi - theta_p))
            elif phi >= np.pi - theta_p and d < 0:
                theta_x = theta * ((np.pi - phi) / theta_p)
            elif phi < np.pi - theta_p and d < 0:
                theta_x = theta * (phi / (np.pi - theta_p))
            rotation_sincos = np.zeros((2, 2))
            rotation_sincos[0][0] = np.cos(theta_x)
            rotation_sincos[0][1] = -np.sin(theta_x)
            rotation_sincos[1][0] = np.sin(theta_x)
            rotation_sincos[1][1] = np.cos(theta_x)
            return np.hstack((np.matmul(rotation_sincos, input_vec), x_p))

        if np.count_nonzero(x) != 0:
            return np.matmul(rotation_matrix.T, rotation(v1, v2, np.matmul(rotation_matrix, x)))
        else:
            return x

    @staticmethod
    def gs_constrained(matrix, v1, v2):
        v1 = np.asarray(v1).reshape(-1)
        v2 = np.asarray(v2).reshape(-1)
        u = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]))
        u[0] = v1
        u[0] = u[0] / np.linalg.norm(u[0])
        u[1] = v2 - proj(u[0], v2)
        u[1] = u[1] / np.linalg.norm(u[1])
        for i in range(0, len(matrix) - 2):
            p = 0.0
            for j in range(0, i + 2):
                p = p + proj(u[j], matrix[i])
            u[i + 2] = matrix[i] - p
            u[i + 2] = u[i + 2] / np.linalg.norm(u[i + 2])
        return u

    @staticmethod
    def correction2d(U, v1, v2, x):
        def rotation(v1, v2, x):
            # v1 = np.asarray(v1).reshape(-1); v2 = np.asarray(v2).reshape(-1); x = np.asarray(x).reshape(-1)
            v2P = U[1]  # basis(np.vstack((v1,v2))); #xP = x[2:len(x)]
            # x = (np.dot(x,v1),np.dot(x,v2P))
            v2 = (np.matmul(v2, v1.T), np.sqrt(1 - (np.matmul(v2, v1.T) ** 2)))
            v1 = (1, 0)
            thetaX = 0.0
            thetaP = np.abs(np.arccos(np.dot(v1, v2)))
            theta = (np.pi / 2.0) - thetaP
            phi = np.arccos(np.dot(v1, x / np.linalg.norm(x)))
            d = np.dot(v2P, x / np.linalg.norm(x))
            if phi < thetaP and d > 0:
                thetaX = theta * (phi / thetaP)
            elif phi > thetaP and d > 0:
                thetaX = theta * ((np.pi - phi) / (np.pi - thetaP))
            elif phi >= np.pi - thetaP and d < 0:
                thetaX = theta * ((np.pi - phi) / thetaP)
            elif phi < np.pi - thetaP and d < 0:
                thetaX = theta * (phi / (np.pi - thetaP))
            R = np.zeros((2, 2))
            R[0][0] = np.cos(thetaX)
            R[0][1] = -np.sin(thetaX)
            R[1][0] = np.sin(thetaX)
            R[1][1] = np.cos(thetaX)
            return np.matmul(R, x)

        if np.count_nonzero(x) != 0:
            return rotation(v1, v2, np.matmul(U, x))
        else:
            return x

    @staticmethod
    def gs_constrained2d(matrix, v1, v2):
        def proj(vec, a):
            return ((np.dot(vec, a.T)) * vec) / (np.dot(vec, vec))

        # v1 = np.asarray(v1).reshape(-1)
        # v2 = np.asarray(v2).reshape(-1)
        u = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]))
        u[0] = v1
        u[0] = u[0] / np.linalg.norm(u[0])
        u[1] = v2 - proj(u[0], v2)
        u[1] = u[1] / np.linalg.norm(u[1])
        return u

    @staticmethod
    def correction2d_new(U, v1, v2, x):
        if np.count_nonzero(x) != 0:
            rotated_x = rotation(v1, v2, x)
            return rotated_x
        else:
            return x

    @staticmethod
    def gs_constrained2d_new(matrix, v1, v2):
        u = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]))
        u[0] = v1
        u[0] = u[0] / np.linalg.norm(u[0])
        u[1] = v2 - proj(u[0], v2)
        u[1] = u[1] / np.linalg.norm(u[1])
        return u


class INLPDebiaser(Debiaser):
    def debias(self, bias_direction, seedwords1, seedwords2, evalwords, num_iters=35):
        y = np.array([0] * len(seedwords1) + [1] * len(seedwords2))
        rowspace_projections = []

        for iter_idx in range(num_iters):
            x_projected = self.debiased_emb.get_vecs(seedwords1 + seedwords2).copy()
            x_eval = self.debiased_emb.get_vecs(evalwords).copy()
            # x_eval = self.debiased_emb.get_vecs(self.base_emb.words()).copy()

            classifier_i = SVC(kernel='linear').fit(x_projected, y)
            weights = np.expand_dims(classifier_i.coef_[0], 0)
            bias_direction = weights[0] / np.linalg.norm(weights[0])

            if (np.linalg.norm(weights) < 1e-10 or classifier_i.score(x_projected, y) < 0.55) and iter_idx > 1:
                break

            # ---------------------------------------------------------
            # Step 1 - PCA of points in their vector space
            # ---------------------------------------------------------
            base_projector = self.animator.add_projector(PCA(n_components=2), name='base_projector')
            base_projector.fit(self.debiased_emb, seedwords1 + seedwords2 + evalwords)

            step1 = self.animator.add_anim_step(camera_step=True)
            step1.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
            step1.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
            step1.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))
            step1.add_points(base_projector.project(self.debiased_emb, [], group=0, direction=bias_direction - bias_direction))

            # ---------------------------------------------------------
            # Step 2 - Identify the best classifier and draw its normal as a unit vector u
            # ---------------------------------------------------------
            step2 = self.animator.add_anim_step()
            step2.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
            step2.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
            step2.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))
            step2.add_points(base_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))
            # classifier to pop into existence

            # ---------------------------------------------------------
            # Step 3 - Project points such that bias direction is aligned with the x-axis
            # ---------------------------------------------------------
            align_projector = self.animator.add_projector(BiasPCA(), name='align_projector')
            align_projector.fit(self.debiased_emb, seedwords1 + seedwords2 + evalwords, bias_direction=bias_direction)

            step3 = self.animator.add_anim_step(camera_step=True)
            step3.add_points(align_projector.project(self.debiased_emb, seedwords1, group=1))
            step3.add_points(align_projector.project(self.debiased_emb, seedwords2, group=2))
            step3.add_points(align_projector.project(self.debiased_emb, evalwords, group=3))
            step3.add_points(align_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))

            # ---------------------------------------------------------
            # Step 4 - Apply linear projection with respect to direction u
            # ---------------------------------------------------------

            p_rowspace_wi = self.get_rowspace_projection(weights)
            rowspace_projections.append(p_rowspace_wi)

            # now project data to new rowspace
            P = self.get_projection_to_intersection_of_nullspaces(rowspace_projections, x_projected.shape[1])
            # x_projected = P.dot(x_projected.T).T
            # x_eval = P.dot(x_eval.T).T

            x_projected = P.dot(self.base_emb.vectors().T).T
            self.debiased_emb.update_vectors(self.base_emb.words(), x_projected)

            # self.debiased_emb.update_vectors(seedwords1 + seedwords2, x_projected)
            # self.debiased_emb.update_vectors(evalwords, x_eval)

            step4 = self.animator.add_anim_step()
            step4.add_points(align_projector.project(self.debiased_emb, seedwords1, group=1))
            step4.add_points(align_projector.project(self.debiased_emb, seedwords2, group=2))
            step4.add_points(align_projector.project(self.debiased_emb, evalwords, group=3))
            step4.add_points(align_projector.project(self.debiased_emb, [], group=0, direction=bias_direction))

        debiased_projector = self.animator.add_projector(PCA(n_components=2), name='debiased_projector')
        debiased_projector.fit(self.debiased_emb, seedwords1 + seedwords2)

        step_final = self.animator.add_anim_step(camera_step=True)
        step_final.add_points(debiased_projector.project(self.debiased_emb, seedwords1, group=1))
        step_final.add_points(debiased_projector.project(self.debiased_emb, seedwords2, group=2))
        step_final.add_points(debiased_projector.project(self.debiased_emb, evalwords, group=3))
        step_final.add_points(debiased_projector.project(self.debiased_emb, [], group=0, direction=bias_direction - bias_direction))

    @staticmethod
    def get_rowspace_projection(W):
        """
        :param W: the matrix over its nullspace to project
        :return: the projection matrix over the rowspace
        """

        if np.allclose(W, 0):
            w_basis = np.zeros_like(W.T)
        else:
            w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

        P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

        return P_W

    def get_projection_to_intersection_of_nullspaces(self, rowspace_projection_matrices, input_dim):
        """
        Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
        this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
        uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
        N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
        :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
        :param dim: input dim
        """

        I = np.eye(input_dim)
        Q = np.sum(rowspace_projection_matrices, axis=0)
        P = I - self.get_rowspace_projection(Q)

        return P


class Projector:
    def __init__(self, projector, name):
        self.projector = projector
        self.name = name

    def fit(self, embedding, words, bias_direction=None, secondary_direction=None):
        if bias_direction is None:
            self.projector.fit(embedding.get_vecs(words))
        else:
            self.projector.fit(embedding.get_vecs(words), bias_direction=bias_direction, secondary_direction=secondary_direction)

    def project(self, embedding, words, group=None, direction=None, concept_idx=1):
        word_vecs_2d = []
        if direction is not None:
            dim = direction.shape[0]
            origin = np.zeros(dim)
            projection = self.projector.transform(np.vstack([origin, direction]))
            projection = projection - projection[0]
            # projection[1] = projection[1] / np.linalg.norm(projection[1])
            words = ['Origin', 'Concept' + str(concept_idx)]
            # words = ['Concept' + str(concept_idx), '']
        else:
            if len(words) != 0:
                projection = self.projector.transform(embedding.get_vecs(words))
            else:
                projection = None

        for i, word in enumerate(words):
            x, y = projection[i][0], projection[i][1]
            word_vecs_2d.append(WordVec2D(word, np.round(x, 6), np.round(y, 6), group=group))

        return word_vecs_2d


class BiasPCA:
    def __init__(self):
        self.vectors = None
        self.bias_direction = None
        self.secondary_direction = None
        self.pca = PCA(n_components=2)

    def fit(self, vectors, bias_direction, secondary_direction=None):
        self.vectors = vectors
        self.bias_direction = bias_direction
        self.secondary_direction = secondary_direction
        if self.secondary_direction is None:  # if not Oscar case
            self.pca.fit(vectors - vectors.dot(self.bias_direction.reshape(-1, 1)) * self.bias_direction)

    def transform(self, vectors):
        debiased_vectors = vectors - vectors.dot(self.bias_direction.reshape(-1, 1)) * self.bias_direction
        x_component = np.expand_dims(vectors.dot(self.bias_direction), 1)
        if self.secondary_direction is None:  # General case where there is no secondary direction
            y_component = np.expand_dims(self.pca.transform(debiased_vectors)[:, 0], 1)
        else:  # Oscar
            y_component = np.expand_dims(vectors.dot(self.secondary_direction), 1)

        return np.hstack([x_component, y_component])


class CoordinateProjector:
    def __init__(self):
        self.vectors = None
        self.bias_direction = None
        self.secondary_direction = None

    def fit(self, vectors, bias_direction, secondary_direction=None):
        self.vectors = vectors
        self.bias_direction = bias_direction
        self.secondary_direction = secondary_direction

    def transform(self, vectors):
        return vectors[:, 0:2]


class WordVec2D:
    def __init__(self, word, x, y, group=None, meta=None):
        eps = 1e-8
        self.label = word
        self.x = x if abs(x) > eps else 0.0
        self.y = y if abs(y) > eps else 0.0
        self.group = group
        self.meta = meta

    def to_dict(self):
        return {'label': self.label, 'x': self.x, 'y': self.y, 'group': self.group}

    def __repr__(self):
        return f'WordVec2D("{self.label}", {self.x}, {self.y}, group={self.group}))'

    def copy(self):
        return WordVec2D(self.label, self.x, self.y, group=self.group, meta=self.meta)


class AnimStep:
    def __init__(self, camera_step=False):
        self.points = []
        self.camera_step = camera_step

    def add_points(self, word_vecs_2d):
        self.points += word_vecs_2d

    def get_point_array(self, filter_groups=[]):
        point_coords = []
        for point in self.points:
            if point.group not in filter_groups:
                point_coords.append([point.x, point.y])
        return np.array(point_coords)


class Animator:
    def __init__(self):
        self.anim_steps = []
        self.transitions = []
        self.projectors = {}

    def add_projector(self, projector, name='GenericProjector'):
        self.projectors[name] = Projector(projector, name)
        return self.projectors[name]

    def add_anim_step(self, camera_step=False):
        new_step = AnimStep(camera_step=camera_step)
        self.anim_steps.append(new_step)
        return new_step

    def make_transition(self, source, target):
        if source is None:
            self.transitions.append(None)

        projection_path = generateFullDynamicProjPath(source.get_point_array(), target.get_point_array())
        transitions = []

        for proj in projection_path:
            transition = []
            for index, vec in enumerate(proj):
                word_vec_copy = source.points[index].copy()
                word_vec_copy.x, word_vec_copy.y = vec[0], vec[1]
                transition.append(word_vec_copy)
            transitions.append(transition)

        self.transitions.append(transitions)

    def convert_animations_to_payload(self):
        payload = []

        for step in self.anim_steps:
            payload.append([point.to_dict() for point in step.points])

        return payload

    def convert_transitions_to_payload(self):
        payload = []

        for transition in self.transitions:
            payload_step = []
            for step in transition:
                payload_step.append([point.to_dict() for point in step])

            payload.append(payload_step)

        return payload

    def get_camera_steps(self):
        camera_steps = []
        for step in self.anim_steps:
            camera_steps.append(step.camera_step)

        return camera_steps

    def get_bounds(self):
        vectors = []

        for step in self.anim_steps:
            for point in step.points:
                vectors.append((point.x, point.y))
        vectors = np.array(vectors)

        return {
            'xmin': vectors[:, 0].min(), 'xmax': vectors[:, 0].max(),
            'ymin': vectors[:, 1].min(), 'ymax': vectors[:, 1].max()
        }


# Various bias direction computation methods
# ----------------------------------------------------
def bias_two_means(embedding, word_list1, word_list2):
    vec1, vec2 = embedding.get_vecs(word_list1), embedding.get_vecs(word_list2)
    vec1_mean, vec2_mean = np.mean(vec1, axis=0), np.mean(vec2, axis=0)
    bias_direction = (vec1_mean - vec2_mean) / np.linalg.norm(vec1_mean - vec2_mean)

    return bias_direction / np.linalg.norm(bias_direction)


def bias_pca(embedding, word_list):
    vecs = embedding.get_vecs(word_list)
    if len(word_list) == 1:
        bias_direction = vecs[0]
    else:
        bias_direction = PCA(n_components=2).fit(vecs).components_[0]

    return bias_direction / np.linalg.norm(bias_direction)


def bias_pca_paired(embedding, pair1, pair2):
    assert len(pair1) == len(pair2)

    if len(pair1) == len(pair2) == 1:
        diff = embedding.get_vecs(pair1) - embedding.get_vecs(pair2)
        return diff[0] / np.linalg.norm(diff[0])

    vec1, vec2 = embedding.get_vecs(pair1), embedding.get_vecs(pair2)
    paired_vecs = vec1 - vec2

    matrix = []

    for a, b in zip(vec1, vec2):
        center = (a + b) / 2
        matrix.append(a - center)
        matrix.append(b - center)

    matrix = np.array(matrix)

    bias_direction = PCA(n_components=2).fit(matrix).components_[0]

    return bias_direction / np.linalg.norm(bias_direction)


def bias_classification(embedding, seedwords1, seedwords2):
    vec1, vec2 = embedding.get_vecs(seedwords1), embedding.get_vecs(seedwords2)
    x = np.vstack([vec1, vec2])
    y = np.concatenate([[0] * vec1.shape[0], [1] * vec2.shape[0]])

    classifier = SVC(kernel='linear').fit(x, y)
    bias_direction = classifier.coef_[0]

    return bias_direction / np.linalg.norm(bias_direction)


def bias_gss(embedding, word_list1, word_list2, num_permutations=2):
    vec1, vec2 = embedding.get_vecs(word_list1), embedding.get_vecs(word_list2)
    vec1, vec2 = np.mean(vec1, axis=0), np.mean(vec2, axis=0)

    if len(word_list2) == 1:
        v = embedding.get(word_list2[0]).vector
        v = np.expand_dims(v, 0)
    else:
        v = np.zeros((num_permutations, vec1.shape[0]))
        for k in range(num_permutations):
            word_list2 = np.random.permutation(word_list2)
            a = embedding.get(word_list2[0]).vector
            b = embedding.get(word_list2[1]).vector

            if len(word_list2) == 2:
                v = gss(f_weat, a, b, vec1, embedding, tol=1e-5)
                v = np.expand_dims(v, 0)
                break
            else:
                for i in range(2, len(word_list2)):
                    a = gss(f_weat, a, b, vec1, embedding, tol=1e-5)
                    b = embedding.get(word_list2[i]).vector
            v[k] = a

    vecB_mean_perm = np.mean(v, 0)

    if len(word_list1) == 1:
        v = embedding.get(word_list1[0]).vector
        v = np.expand_dims(v, 0)
    else:
        v = np.zeros((num_permutations, vec2.shape[0]))
        for k in range(num_permutations):
            word_list1 = np.random.permutation(word_list1)
            a = embedding.get(word_list1[0]).vector
            b = embedding.get(word_list1[1]).vector

            if len(word_list1) == 2:
                v = gss(f_weat, a, b, vecB_mean_perm, embedding, tol=1e-5)
                v = np.expand_dims(v, 0)
                break
            else:
                for i in range(2, len(word_list1)):
                    a = gss(f_weat, a, b, vecB_mean_perm, embedding, tol=1e-5)
                    b = embedding.get(word_list1[i]).vector
            v[k] = a

    vecA_mean_perm = np.mean(v, 0)

    bias_direction = (vecA_mean_perm - vecB_mean_perm) / np.linalg.norm(vecA_mean_perm - vecB_mean_perm)
    return bias_direction / np.linalg.norm(bias_direction)


def get_bias_direction(embedding, seedwords1, seedwords2, subspace_method):
    if subspace_method == 'Two-means':
        bias_direction = bias_two_means(embedding, seedwords1, seedwords2)
    elif subspace_method == 'PCA':
        bias_direction = bias_pca(embedding, seedwords1)
    elif subspace_method == 'PCA-paired':
        bias_direction = bias_pca_paired(embedding, seedwords1, seedwords2)
    elif subspace_method == 'Classification':
        bias_direction = bias_classification(embedding, seedwords1, seedwords2)
    elif subspace_method == 'GSS':
        bias_direction = bias_gss(embedding, seedwords1, seedwords2)
    else:
        raise ValueError('Incorrect subspace method')

    return bias_direction / np.linalg.norm(bias_direction)


# IO helpers
def read_embeddings(path, limit=100000):
    word_vectors = {}

    with open(path, 'r', encoding='UTF-8') as vec_file:
        # for idx, line in enumerate(vec_file):
        for idx, line in tqdm(enumerate(vec_file), total=limit, unit_scale=True):
            line = line.rstrip().split()
            word = line[0]
            vector = np.array(line[1:]).astype('float')
            vector = vector / np.linalg.norm(vector)
            word_vectors[word] = WordVector(idx, word, vector)

            if idx >= limit - 1:
                break

    return word_vectors


def read_merchant_embedding(emb_path, metadata_path, group, limit=20000, normalize=True):
    word_vectors = {}
    metadata = pd.read_csv(metadata_path, delimiter='\t', names=['merch_id', 'merch_name', 2, 3, 4, 5, 6], index_col='merch_id', header=None)

    with open(emb_path, 'r') as vec_file:
        for idx, line in tqdm(enumerate(vec_file)):
            if idx == 0:
                continue

            line = line.rstrip().split()
            merch_name = metadata.loc[int(line[0])].merch_name
            vector = np.array(line[1:]).astype('float')
            if normalize:
                vector = vector / np.linalg.norm(vector)

            word_vectors[merch_name] = WordVector(merch_name, vector, group)

            if idx >= limit - 1:
                break

    return word_vectors


def save(savepath, obj):
    with open(savepath, 'wb') as savefile:
        pickle.dump(obj, savefile)


def load(loadpath):
    with open(loadpath, 'rb') as loadfile:
        return pickle.load(loadfile)


# Misc
def basis(vec):
    first_component = vec[0]
    second_component = vec[1]
    v2_prime = second_component - first_component * float(np.matmul(first_component, second_component.T))
    v2_prime = v2_prime / np.linalg.norm(v2_prime)
    return v2_prime


def proj(u, a):
    return (np.dot(u, a)) * u


def rotation(v1, v2, x):
    v2P = v2 - proj(v1, v2)
    v2P = v2P / np.linalg.norm(v2P)

    thetaP = np.arccos(np.dot(v1, v2))
    theta = np.abs(thetaP - np.pi / 2)

    x_norm = x / np.linalg.norm(x)
    phi = np.arccos(np.dot(v1 / np.linalg.norm(v1), x_norm))
    d = np.dot(v2P, x_norm)

    if d > 0 and phi < thetaP:
        thetaX = theta * (phi / thetaP)
    elif d > 0 and phi > thetaP:
        thetaX = theta * ((np.pi - phi) / (np.pi - thetaP + 1e-10))
    elif d < 0 and phi >= np.pi - thetaP:
        thetaX = theta * ((np.pi - phi) / thetaP)
    elif d < 0 and phi < np.pi - thetaP:
        thetaX = theta * (phi / (np.pi - thetaP + 1e-10))
    else:
        return x

    R = np.zeros((2, 2))
    R[0][0] = np.cos(thetaX)
    R[0][1] = -np.sin(thetaX)
    R[1][0] = np.sin(thetaX)
    R[1][1] = np.cos(thetaX)

    return np.matmul(R, x)


def two_means(embedding, word_list1, word_list2):
    vec1, vec2 = embedding.get_many(word_list1), embedding.get_many(word_list2)
    vec1_mean, vec2_mean = np.mean(vec1, axis=0), np.mean(vec2, axis=0)

    bias_direction = (vec1_mean - vec2_mean) / np.linalg.norm(vec1_mean - vec2_mean)
    return vec1_mean, vec2_mean, bias_direction


def compute_weat_score(embedding, X, Y, A, B):
    X_vecs, Y_vecs, A_vecs, B_vecs = [embedding.get_vecs(wordlist) for wordlist in [X, Y, A, B]]
    return weat_score(X_vecs, Y_vecs, A_vecs, B_vecs)


def debias_linear_projection(embedding, bias_vec):
    debiased = embedding.vectors - embedding.vectors.dot(bias_vec.reshape(-1, 1)) * bias_vec
    return debiased


def hard_debias_get_bias_direction(embedding: Embedding, word_list1: list, word_list2: list, n_components: int = 10):
    matrix = []
    for w1, w2 in zip(word_list1, word_list2):
        center = (embedding.get(w1) + embedding.get(w2)) / 2
        matrix.append(embedding.get(w1) - center)
        matrix.append(embedding.get(w2) - center)

    matrix = np.array(matrix)
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    return pca.components_[0]


def remove_component(u, v):
    return u - v * u.dot(v) / v.dot(v)


def hard_debias(base_embedding: Embedding, debiased_embedding: Embedding, bias_vec: np.ndarray, eval_words: list):
    eval_wordset = set(eval_words)
    debiased_embedding.vectors = []
    for i, word in enumerate(base_embedding.words):
        # remove bias direction from evaluation words
        if word in eval_wordset:
            debiased_embedding.vectors.append(remove_component(base_embedding.vectors[i], bias_vec))
        else:
            debiased_embedding.vectors.append(base_embedding.vectors[i])

    debiased_embedding.vectors = np.array(debiased_embedding.vectors)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# If called directly from command line, create and save Embedding object
if __name__ == '__main__':
    # dirty hack to make sure the object can be unpickled in the flask app
    # noinspection PyUnresolvedReferences
    from vectors import Embedding

    # datapath = './data/Embedding_Restaurant/bay_filter.emb'
    # metadata_path = './data/Embedding_Restaurant/bay_filter'
    # emb_bay = Embedding(datapath, metadata_path, 0)
    #
    # datapath = './data/Embedding_Restaurant/la_filter.emb'
    # metadata_path = './data/Embedding_Restaurant/la_filter'
    # emb_la = Embedding(datapath, metadata_path, 1)
    #
    # emb_bay.merge(emb_la)
    #
    # save('data/bayarea_la.pkl', emb_bay)

    emb = Embedding('data/common_crawl/glove.42B.300d.txt', limit=100000)
    save('data/common_crawl.pkl', emb)

    emb = Embedding('data/wiki_gigaword/glove.6B.300d.txt', limit=100000)
    save('data/wiki_gigaword.pkl', emb)

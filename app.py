import os

from flask import Flask, render_template, request, jsonify, send_file
from vectors import *
import utils
from vectors import get_bias_direction
import json
import copy
import csv

app = Flask(__name__)

app.base_embedding = load('data/embedding.pkl')
app.debiased_embedding = load('data/embedding.pkl')
app.frozen = False

with open('static/assets/explanations.json', 'r') as explanation_json:
    app.explanations = json.load(explanation_json)

# app.weat_A = ['doctor', 'engineer', 'lawyer', 'mathematician', 'banker']
# app.weat_B = ['receptionist', 'homemaker', 'nurse', 'dancer', 'maid']

# app.debiased_embedding.word_vectors = app.base_embedding.word_vectors.copy()

ALGORITHMS = {
    'Algorithm: Linear projection': 'Linear',
    'Algorithm: Hard debiasing': 'Hard',
    'Algorithm: OSCaR': 'OSCaR',
    'Algorithm: Iterative Null Space Projection': 'INLP'
}

SUBSPACE_METHODS = {
    'Subspace method: Two means': 'Two-means',
    'Subspace method: PCA': 'PCA',
    'Subspace method: PCA-paired': 'PCA-paired',
    'Subspace method: Classification': 'Classification',
    'Subspace method: GSS': 'GSS'
}


def reload_embeddings():
    print('Reloaded embedding')
    app.base_embedding = load('data/embedding.pkl')
    app.debiased_embedding = load('data/embedding.pkl')  # Embedding(None)
    # app.debiased_embedding.word_vectors = app.base_embedding.word_vectors.copy()


def rename_concepts(anim_steps, c1_name, c2_name):
    for step in anim_steps:
        for point in step:
            if point['label'] == 'Concept1':
                point['label'] = c1_name.replace(' ', '_')
            if point['label'] == 'Concept2':
                point['label'] = c2_name.replace(' ', '_')


@app.route('/')
def index():
    return render_template('interface.html')


@app.route('/seedwords2', methods=['POST'])
def get_seedwords2():
    try:
        if not app.frozen:
            reload_embeddings()

        seedwords1, seedwords2, evalwords = request.values['seedwords1'], request.values['seedwords2'], request.values['evalwords']
        equalize_set = request.values['equalize']
        orth_subspace_words = request.values['orth_subspace']
        concept1_name, concept2_name = request.values['concept1_name'], request.values['concept2_name']

        algorithm, subspace_method = ALGORITHMS[request.values['algorithm']], SUBSPACE_METHODS[request.values['subspace_method']]

        seedwords1 = utils.process_seedwords(seedwords1)
        seedwords2 = utils.process_seedwords(seedwords2)
        evalwords = utils.process_seedwords(evalwords)
        equalize_set = [list(map(lambda x: x, word.split('-'))) for word in utils.process_seedwords(equalize_set)][:5]

        orth_subspace_words = utils.process_seedwords(orth_subspace_words)

        if subspace_method == 'PCA-paired':
            seedwords1, seedwords2 = list(zip(*[(w.split('-')[0], w.split('-')[1]) for w in seedwords1]))

        if subspace_method == 'PCA':
            seedwords2 = []

        # Perform debiasing according to algorithm and subspace direction method
        bias_direction = get_bias_direction(app.base_embedding, seedwords1, seedwords2, subspace_method)

        explanations = app.explanations
        # weatscore_predebiased = utils.get_weat_score(app.base_embedding, app.weat_A, app.weat_B)
        # weatscore_postdebiased = utils.get_weat_score(app.debiased_embedding, app.weat_A, app.weat_B)

        if algorithm == 'Linear':
            debiaser = LinearDebiaser(app.base_embedding, app.debiased_embedding, app)
            debiaser.debias(bias_direction, seedwords1, seedwords2, evalwords)

        elif algorithm == 'Hard':
            debiaser = HardDebiaser(app.base_embedding, app.debiased_embedding, app)
            debiaser.debias(bias_direction, seedwords1, seedwords2, evalwords, equalize_set=equalize_set)

        elif algorithm == 'OSCaR':
            debiaser = OscarDebiaser(app.base_embedding, app.debiased_embedding, app)
            debiaser.debias(bias_direction, seedwords1, seedwords2, evalwords, orth_subspace_words, bias_method=subspace_method)

        elif algorithm == 'INLP':
            debiaser = INLPDebiaser(app.base_embedding, app.debiased_embedding, app)
            debiaser.debias(bias_direction, seedwords1, seedwords2, evalwords)
            explanations['INLP'] += explanations['INLP'][1:5] * (len(debiaser.animator.anim_steps) // 5)

        anim_steps = debiaser.animator.convert_animations_to_payload()
        transitions = debiaser.animator.convert_transitions_to_payload()
        rename_concepts(anim_steps, concept1_name, concept2_name)

        data_payload = {'base': anim_steps[0],
                        'debiased': anim_steps[-1],
                        'anim_steps': anim_steps,
                        'transitions': transitions,
                        'bounds': debiaser.animator.get_bounds(),
                        'explanations': explanations[algorithm],
                        'camera_steps': debiaser.animator.get_camera_steps(),
                        # 'weat_scores': {'pre-weat': weatscore_predebiased, 'post-weat': weatscore_postdebiased}
                        }

        return jsonify(data_payload)
    except KeyError as e:
        print(e)
        raise InvalidUsage(f'Something went wrong! Could not find the word {str(e).strip()}', 404)


@app.route('/freeze', methods=['POST'])
def freeze_embedding():
    app.frozen = True
    app.base_embedding = copy.deepcopy(app.debiased_embedding)
    return jsonify('Updated')


@app.route('/unfreeze', methods=['POST'])
def unfreeze_embedding():
    app.frozen = False
    reload_embeddings()
    return jsonify('Updated')


@app.route('/export')
def export_as_csv():
    filepath = 'static/upload/debiased.csv'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(app.debiased_embedding.to_csv_list())

    return send_file(filepath, attachment_filename='debiased.csv', mimetype='text/csv')


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

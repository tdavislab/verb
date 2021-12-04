import os
from os.path import isfile, join
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.neighbors import NearestNeighbors

from vectors import *
import utils
from vectors import get_bias_direction
import json
import copy
import csv

app = Flask(__name__)
app.embedding_path = 'data/embedding.pkl'
app.base_embedding = load(app.embedding_path)
app.debiased_embedding = load(app.embedding_path)
app.frozen = False
app.base_knn = None
app.debiased_knn = None

# for parallel coordinates
language = "en"
df, model = None, None

with open('static/assets/explanations.json', 'r') as explanation_json:
    app.explanations = json.load(explanation_json)

app.weat_A = ['doctor', 'engineer', 'lawyer', 'mathematician', 'banker']
app.weat_B = ['receptionist', 'homemaker', 'nurse', 'dancer', 'maid']
app.male_words = ['man', 'male', 'boy', 'brother', 'him', 'his', 'son']
app.female_words = ['woman', 'female', 'girl', 'brother', 'her', 'hers', 'daughter']

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
    app.base_embedding = load(app.embedding_path)
    app.debiased_embedding = load(app.embedding_path)  # Embedding(None)
    # app.debiased_embedding.word_vectors = app.base_embedding.word_vectors.copy()


app.reload_embeddings = reload_embeddings


def rename_concepts(anim_steps, c1_name, c2_name):
    for step in anim_steps:
        for point in step:
            if point['label'] == 'Concept1':
                point['label'] = c1_name.replace(' ', '_')
            if point['label'] == 'Concept2':
                point['label'] = c2_name.replace(' ', '_')


def compute_knn(k=11):
    base_words, base_vecs = app.base_embedding.words(), app.base_embedding.vectors()
    app.base_knn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(base_vecs)
    debiased_words, debiased_vecs = app.debiased_embedding.words(), app.debiased_embedding.vectors()
    app.debiased_knn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(debiased_vecs)


def neighbors(embedding, knn_obj, word_list):
    vecs = embedding.get_vecs(word_list)
    neighbor_indices = knn_obj.kneighbors(vecs, return_distance=False)
    words = embedding.words()
    return {word: [words[i] for i in neighbor_indices[idx]] for idx, word in enumerate(word_list)}


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
            seedwords1 = list(seedwords1)
            seedwords2 = list(seedwords2)

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

        compute_knn()
        if algorithm == 'Hard':
            word_list = seedwords1 + seedwords2 + evalwords + orth_subspace_words + [y for x in equalize_set for y in x]
        else:
            word_list = seedwords1 + seedwords2 + evalwords + orth_subspace_words

        word_list = list(set([w for w in word_list if not (w == '' or w == [''])]))
        base_neighbors = neighbors(app.base_embedding, app.base_knn, word_list)
        debiased_neighbors = neighbors(app.debiased_embedding, app.debiased_knn, word_list)

        # base_neighbors = {word: ['t'] for word in word_list}
        # debiased_neighbors = {word: ['t'] for word in word_list}

        data_payload = {'base': anim_steps[0],
                        'debiased': anim_steps[-1],
                        'anim_steps': anim_steps,
                        'transitions': transitions,
                        'bounds': debiaser.animator.get_bounds(),
                        'explanations': explanations[algorithm],
                        'camera_steps': debiaser.animator.get_camera_steps(),
                        'knn': {'base': base_neighbors, 'debiased': debiased_neighbors}
                        }
        return jsonify(data_payload)
    except KeyError as e:
        print(e)
        raise InvalidUsage(f'Something went wrong! Could not find the word {str(e).strip()}', 404)
    except Exception as e:
        raise InvalidUsage(f'Something went wrong! Error message from the backend: \n{str(e).strip()}', 404)


@app.route('/freeze', methods=['POST'])
def freeze_embedding():
    app.frozen = True
    app.base_embedding = copy.deepcopy(app.debiased_embedding)
    return jsonify('Updated')


@app.route('/unfreeze', methods=['GET'])
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


@app.route('/import', methods=['POST'])
def import_csv():
    filepath = 'data/imported_embedding.pkl'


@app.route('/weat', methods=['POST'])
def get_weat():
    weat_a, weat_b = request.values['occupation_a'], request.values['occupation_b']
    male_words, female_words = request.values['gender_a'], request.values['gender_b']
    weat_a, weat_b = utils.process_seedwords(weat_a), utils.process_seedwords(weat_b)
    male_words, female_words = utils.process_seedwords(male_words), utils.process_seedwords(female_words)

    weatscore_predebiased = utils.get_weat_score(app.base_embedding, weat_a, weat_b, male_words, female_words)
    weatscore_postdebiased = utils.get_weat_score(app.debiased_embedding, weat_a, weat_b, male_words, female_words)

    return jsonify(weat_scores={'pre-weat': weatscore_predebiased, 'post-weat': weatscore_postdebiased})


@app.route('/save_example', methods=['POST'])
def save_example():
    example = request.values.to_dict()

    with open('./static/assets/user_examples.json', 'r+') as user_examples:
        curr_data = json.load(user_examples)
        curr_data['data'].append(example)
        user_examples.seek(0)
        json.dump(curr_data, user_examples, indent=2)
        user_examples.truncate()

    return jsonify('Success')


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


# Adding code for Parallel axis


@app.route('/set_model')
def set_model():
    name = request.args.get("embedding")
    load_embedding(name)
    return "success"


def load_embedding(name):
    global model, language
    if model is None:
        name = "Word2Vec"
        #name = "Glove (wiki 300d)" 
    print("Embedding name: ", name)
    if name=="Word2Vec":
        language = 'en'
        model =  word2vec.KeyedVectors.load_word2vec_format('./data/word_embeddings/word2vec_50k.bin', binary=True, limit=50041) 
    elif name=="Glove (wiki 300d)":
        # print("Glove word embedding backend")
        language = 'en'
        model = KeyedVectors.load_word2vec_format('./data/word_embeddings/glove_50k.bin', binary=True) #   
    elif name=="Word2Vec debiased":
        # print('./data/word_embeddings/GoogleNews-vectors-negative300-hard-debiased.bin')
        language = 'en'
        model = KeyedVectors.load_word2vec_format('./data/word_embeddings/GoogleNews-vectors-negative300-hard-debiased.bin', binary=True, limit=50000) 
    return

@app.route('/get_csv/')
def get_csv():
    global df
    scaling = request.args.get("scaling")
    embedding = request.args.get("embedding")
    if embedding is None:
        embedding = "Word2Vec"

    print("/get_csv/")
    print("Scaling: ", scaling)
    print("Embedding: ", embedding)
    
    if embedding=="Word2Vec":
        if scaling=="Normalization":
            df = pd.read_csv("./data/word2vec_50k.csv",header=0, keep_default_na=False)
        elif scaling=="Percentile":
            df = pd.read_csv("./data/word2vec_50k_percentile.csv",header=0, keep_default_na=False)
        else:
            df = pd.read_csv("./data/word2vec_50k_raw.csv",header=0, keep_default_na=False)
    elif embedding=="Glove (wiki 300d)":
        if scaling=="Normalization":
            df = pd.read_csv("./data/glove_50k.csv",header=0, keep_default_na=False)
        elif scaling=="Percentile":
            df = pd.read_csv("./data/glove_50k_percentile.csv",header=0, keep_default_na=False)
    out = df.to_json(orient='records')
    #print("out", out)
    return out

@app.route('/get_all_words')
def get_all_words():
    if not model:
        setModel()
    return jsonify(list(model.vocab.keys()))


@app.route('/fetch_data',methods=['POST'])
def fetch_data():
    json_data = request.get_json(force=True);       
    slider_sel = json_data['slider_sel']
    hist_type = json_data["hist_type"]
    #print("fetch_data: ", json_data["data"])
    df = pd.json_normalize(json_data['data'])
    #col_list = list(bias_words.keys())
    col_list = [c for c in df.columns if c!="word"]
    # histogram type - ALL, gender, race, eco
    filter_column = None
    if hist_type=="ALL":
        filter_column = df[col_list].abs().mean(axis=1)
    else:
        filter_column = df[hist_type]

    # print("Slider selection: ", slider_sel)
    # list of selected index based on selection
    ind = pd.Series([False]*df.shape[0])
    for slider in slider_sel:
        minV = slider[0]
        maxV = slider[1]
        if (minV != maxV):
            ind = ind | ((filter_column >= minV) & (filter_column <= maxV))

    # print("selected dataframe: ")
    col_list = ["word"] + col_list
    out = df.loc[ind, col_list].to_json(orient='records')
    return jsonify(out)

@app.route('/getFileNames/')
def getFileNames():
    #gp_path, tar_path, word_sim, word_ana = None, None, None, None
    gp_path, tar_path = None, None
    if language=='hi':
        gp_path = './data/wordList/groups/hi/'
        tar_path = './data/wordList/target/hi/'
        #word_sim = './data/benchmark/word_similarity/hi/'
        #word_ana = './data/benchmark/word_analogy/hi/'
    elif language=='fr':
        gp_path = './data/wordList/groups/fr/'
        tar_path = './data/wordList/target/fr/'
        #word_sim = './data/benchmark/word_similarity/fr/'
        #word_ana = './data/benchmark/word_analogy/fr/'
    else:
        gp_path = './data/wordList/groups/en/'
        tar_path = './data/wordList/target/en/'
        #word_sim = './data/benchmark/word_similarity/en/'
        #word_ana = './data/benchmark/word_analogy/en/'
    #target = os.listdir(tar_path)
    target_files = [f for f in os.listdir(tar_path) if isfile(join(tar_path, f))]
    #group = os.listdir(gp_path)
    group_files = [f for f in os.listdir(gp_path) if isfile(join(gp_path, f))]
    #sim_files = os.listdir(word_sim)
    #ana_files = os.listdir(word_ana)
    #return jsonify([group,target,sim_files,ana_files])
    return jsonify([group_files,target_files])

# populate default set of target words
@app.route('/getWords/')
def getWords():
    path = request.args.get("path")
    words = []
    f = open(path, "r", encoding="utf8")
    for x in f:
        if len(x)>0:
            x = x.strip().lower()
            words.append(x)
    return jsonify({"target":words})

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

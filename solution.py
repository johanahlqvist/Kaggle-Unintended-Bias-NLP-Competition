from datetime import datetime
import re

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import Sequence
from tensorflow.keras.losses import binary_crossentropy
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from collections import Counter, defaultdict
from sortedcollections import ValueSortedDict


EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
CHAR_EMBEDDING_FILE = '../input/fasttext-300d-subword/crawl-300d-2M-subword.vec'
FOUL_WORD_DROPOUT = 0
VAL_FRAC = 0
BATCH_SIZE = 256
LSTM_UNITS = 128
N_WORD_DIMENSIONS = 300
EPOCHS = 4
NUM_MODELS = 2
TRAIN_LAST = True
DENSE_ENSAMBLE = False
MAX_LEN = 220
REAL_IDENTITY_COLUMNS = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'

TOXICITY_TYPES = ['insult', 'severe_toxicity', 'obscene', 'identity_attack', 'threat', 'sexual_explicit']
SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

EMBEDDING_DTYPE = 'float32'
USE_BN = False
FTM_WORDS = 3
FTM_CHANNELS = 16
L2_REGULARIZATION = 0
WORD_DROPOUT = 0.2
DENSE_DROPOUT = 0
FOUL_FREQ_TO_DROP = 4
MIN_FOUL_COUNT = 100
MODEL_NAME = 'our_model'
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'
#TOKEN_FILTER = [".", ",", "\n", "\'", "\"", "-", "?", "/", "!", ")", "(", ":", "$", "%", ";", "’", "_", "&", "“", "”", "=", "*", "\xad", "+", "#", ">", "[", "]", "—", "…", "–", "`", "@", "<", "é", "‘", "ᴇ", "ᴏ", "ɴ", "ᴵ", "~", "\u2004", "ᴛ", "\t", "ᴀ", "\xa0", "ʀ", "ɪ", "ʜ", "•", "ᴍ", "▀", "ᴜ", "ᴄ", "ʏ", "ʟ", "ᴅ", "^", "▄", "}"]
TOKEN_FILTER = '\''

class Attention(tensorflow.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tensorflow.keras.layers.Dense(units)
        self.W2 = tensorflow.keras.layers.Dense(units)
        self.V = tensorflow.keras.layers.Dense(1)


    def call(self, inputs):
        features = inputs[0]
        hidden = inputs[1]
        hidden_with_time_axis = tensorflow.expand_dims(hidden, 1)
        score = tensorflow.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tensorflow.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tensorflow.reduce_sum(context_vector, axis=1)
        return [context_vector, attention_weights]

        
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype=EMBEDDING_DTYPE)


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)
        

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300), dtype=EMBEDDING_DTYPE)
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            try:
                embedding_matrix[i] = embedding_index[word.lower()]
            except KeyError:
                pass
    return embedding_matrix


def build_model(embedding_matrix, num_aux_targets, w, losses, l2_reg=0, embedding_trainable=False):
    if l2_reg > 0:
        reg = l2(l2_reg)
    else:
        reg = None
    raw_comments = Input(shape=(MAX_LEN,), name='comment_input')
    
    word_vectors = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                             weights=[embedding_matrix],
                             trainable=embedding_trainable)(raw_comments)
    x = SpatialDropout1D(rate=WORD_DROPOUT)(word_vectors)

    lstm, forward_h, forward_c, backward_h, backward_c  = Bidirectional(CuDNNLSTM(LSTM_UNITS, kernel_regularizer=reg, return_sequences=True, return_state=True, name='sentence_lstm'), name='bidirectional')(x)
    lstm_out = Activation('relu', name='sentence_lstm_relu')(lstm)
    state_h = Concatenate(name='sentence_lstm_state_conc')([forward_h, backward_h])
    attention_context, attention_weights = Attention(MAX_LEN)([lstm_out, state_h])

    comment_stats = Input(shape=(7,), name='comment_stats')

    d1 = dense_block(attention_context, 'lstm_dense', reg=reg)
    
    x = Concatenate(name='gather_for_pred_conc')([d1, comment_stats])
    x = BatchNormalization()(x)
    #x = dense_block(fc_input, 'gather_block', reg=reg)
    last_out = Dense(int((x.get_shape().as_list()[1])), kernel_regularizer=reg, name='summarizer')(x)
    
    x = Dense(1, kernel_regularizer=reg, name='toxicity_dense')(last_out)
    pred = Activation('sigmoid', name='toxicity')(x)
    
    x = Dense(num_aux_targets, kernel_regularizer=reg, name='toxicity_type_dense')(last_out)
    toxicity_type = Activation('softmax', name='toxicity_type')(x)    
    
    #x = Dense(10, kernel_regularizer=reg, name='identity_dense')(last_out)
    #identity_type = Activation('softmax', name='identity')(x)  

    x = Dense(1, kernel_regularizer=reg, name='toxic_identity_dense')(last_out)
    toxic_identity = Activation('sigmoid', name='toxic_identity')(x)  
    
    #model = Model(inputs=[raw_comments, comment_stats], outputs=[pred, toxicity_type, identity_type, toxic_identity])
    model = Model(inputs=[raw_comments, comment_stats], outputs=[pred, toxicity_type, toxic_identity])
    model.compile(optimizer='nadam',
                  loss=losses, loss_weights=w, metrics=['acc'])
    print(model.summary())
    return model

def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]

def dense_block(x_in, nbr, reg=l2(0.01)):
    layers = [(x_in.get_shape().as_list()[1])/2, int(x_in.get_shape().as_list()[1]/4), int(x_in.get_shape().as_list()[1]/8)]
    
    if DENSE_DROPOUT > 0:
        x_in = Dropout(rate=DENSE_DROPOUT)(x_in)
    
    x = x_in

    counter = 0
    for layer_size in layers:
        x = Dense(layer_size, kernel_regularizer=reg, name='dense-block-' + nbr + '-' + str(counter))(x)
        x = Activation('relu', name='dense-block-relu-' + nbr + '-' + str(counter))(x)
        if USE_BN:
            x = BatchNormalization(name='bn-' + nbr + '-' + str(counter))(x)
        counter += 1

    skip_connection = Dense(layers[-1], kernel_regularizer=reg, name='dense-block-skip-connection-' + nbr)(x_in)
    output = Concatenate(name='dense-block-conc-' + nbr)([x, skip_connection])
    output = Activation('relu', name='dense-block-relu' + str(nbr))(output)
    return output
    
    
def conv_block(x_in, kernel, nbr, reg=l2(0.01)):
    layers = [int((x_in.get_shape().as_list()[-1])/2), int(x_in.get_shape().as_list()[-1]/4), int(x_in.get_shape().as_list()[-1]/8)]
    
    if DENSE_DROPOUT > 0:
        x_in = Dropout(rate=DENSE_DROPOUT)(x_in)
    
    x = x_in

    counter = 0
    for layer_size in layers:
        x = Conv1D(layer_size, kernel[counter], kernel_regularizer=reg, name='conv-block-' + nbr + '-' + str(counter), padding='same')(x)
        x = Activation('relu', name='conv-block-relu-' + nbr + '-' + str(counter))(x)
        if USE_BN:
            x = BatchNormalization(name='bn-' + nbr + '-conv-block-' + str(counter))(x)
        counter += 1

    skip_connection = Conv1D(layers[-1], kernel[-1], kernel_regularizer=reg, name='conv-block-skip-connection-' + nbr, padding='same')(x_in)
    output = Concatenate(name='conv-block-conc-' + nbr)([x, skip_connection])
    output = Activation('relu', name='conv-block-relu' + str(nbr))(output)
    return output
    
def core_model(embedding_matrix):
    if L2_REGULARIZATION > 0:
        reg = l2(L2_REGULARIZATION)
    else:
        reg = None
    raw_comments = Input(shape=(MAX_LEN,), name='comment_input')
    word_vectors = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                     weights=[embedding_matrix],
                     trainable=False)(raw_comments)
    x = SpatialDropout1D(rate=WORD_DROPOUT)(word_vectors)
    
    lstm, forward_h, forward_c, backward_h, backward_c  = Bidirectional(CuDNNLSTM(LSTM_UNITS, kernel_regularizer=reg, return_sequences=True, return_state=True, name='sentence_lstm'), name='bidirectional')(x)
    lstm_out = Activation('relu', name='sentence_lstm_relu')(lstm)
    state_h = Concatenate(name='sentence_lstm_state_conc')([forward_h, backward_h])
    
    attention_context, attention_weights = Attention(MAX_LEN)([lstm_out, state_h])


    d1 = dense_block(attention_context, 'lstm_dense', reg=reg)
    stats = Input(shape=(7,), name='comment_stats')

    x = Concatenate(name='gather_for_pred_conc')([d1, stats])

    last_out = Dense(int((x.get_shape().as_list()[1])), kernel_regularizer=reg, name='summarizer')(x)
    
    x = Dense(1, kernel_regularizer=reg, name='toxicity_dense')(last_out)
    pred = Activation('sigmoid', name='toxicity')(x) # detta vill vi nog ha
    
    model = Model(inputs=[raw_comments, stats], outputs=pred)
    return model
    
def ensemble(embedding_matrix, models, embedding_trainable = False):
    if L2_REGULARIZATION > 0:
        reg = l2(L2_REGULARIZATION)
    else:
        reg = None
    raw_comments = Input(shape=(MAX_LEN,), name='comment_input')

    stats = Input(shape=(7,), name='comment_stats')
    
    preds = []
    for m in models:
        cm = core_model(embedding_matrix)
        for layer in cm.layers:
            try:
                layer.set_weights(m[layer.name])
            except KeyError:
                if 'attention' in layer.name:
                    for name, weight in m.items():
                        if 'attention' in name:
                            layer.set_weights(m[name])
                elif 'embedding' in layer.name:
                    for name, weight in m.items():
                        if 'embedding' in name:
                            layer.set_weights(m[name])
                else:
                    print('Unset weights for layer as layer not found: %s' % layer.name)
            
        cm.trainable = False
        preds.append(cm([raw_comments, stats]))
    
    preds = Concatenate()(preds)
    preds = Dropout(0.3)(preds)
    pred = Dense(1)(preds)
    pred = Activation('sigmoid')(pred)
    model = Model(inputs=[raw_comments, stats], outputs=pred)
    print(model.summary())
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['acc'])
    return model
    
    
# Copied from the benchmark kernel
def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])


def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])


def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])


def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)
    

# Convert taget and identity columns to booleans
def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)
    
    
def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    columns = ['target'] + IDENTITY_COLUMNS + TOXICITY_TYPES
    for col in columns:
        convert_to_bool(bool_df, col)
    return bool_df

    
def get_stats(row, cap_re, factor=100):
    # Some factor to make this data more pronounced
    return [row['comment_text'].count('!')/len(row['comment_text'])*factor, row['comment_text'].count('?')/len(row['comment_text'])*factor, 
            row['comment_text'].count(',')/len(row['comment_text'])*factor, row['comment_text'].count('.')/len(row['comment_text'])*factor,
            row['comment_text'].count('\n')/len(row['comment_text'])*factor, row['comment_text'].count('-')/len(row['comment_text'])*factor,
            len(re.sub(cap_re, '', row['comment_text']))/len(row['comment_text'])*factor]


def add_comment_stats(df):
    cap_re = re.compile('[^A-Z]')
    df['stats'] = df.apply(lambda row: get_stats(row, cap_re), axis=1)
    return df
    
    
def identity_words(df):
    # Analyze words, derive stats 

    good_counter = Counter()
    good_count = 0
    for batch in range(int(len(df)/100000)):
        sum_identities = np.sum(df.iloc[(batch)*100000: (batch+1)*100000][REAL_IDENTITY_COLUMNS].values, axis=1).astype('bool')
        good_comments = ' '.join(df.iloc[(batch)*100000: (batch+1)*100000][sum_identities == 0].apply(lambda row: row['comment_text'], axis=1)).split(' ')
        good_counter.update(good_comments)
        good_count += len(good_comments)
        del good_comments
    
    sum_identities = np.sum(df[REAL_IDENTITY_COLUMNS].values, axis=1).astype('bool')
    foul_comments = ' '.join(df[sum_identities > 0].apply(lambda row: row['comment_text'], axis=1)).split(' ')
    foul_counter = Counter(foul_comments)
    foul_count = len(foul_comments)
    del foul_comments
    
    word_foulness = ValueSortedDict()
    for word, count in good_counter.most_common() + foul_counter.most_common():
        if count > MIN_FOUL_COUNT:
            bad_freq = foul_counter[word]/foul_count
            good_freq = (good_counter[word] + 1)/good_count
            bad_prob = bad_freq/(good_freq)
            word_foulness[word + '_'] = bad_prob

    # Because of a weird bug we had to add an underscore to every word, let's remove it
    foul_words = [word[:-1] for word in list(word_foulness.islice(word_foulness.bisect_key(FOUL_FREQ_TO_DROP)))]
    return foul_words
    
    
def check_auc(df_val, preds):
    df_val[MODEL_NAME] = preds
    bias_metrics_df = compute_bias_metrics_for_model(df_val, IDENTITY_COLUMNS, MODEL_NAME, TOXICITY_COLUMN)
    final_metric = get_final_metric(bias_metrics_df, calculate_overall_auc(df_val, MODEL_NAME))
    print(bias_metrics_df)
    print(final_metric)
    

    
def log(msg):
    print('[%s]: %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
    
def separate_tokens(row):
    row['comment_text'] = re.sub(r'([a-zA-Z0-9])([^a-zA-Z0-9\'´` ])', r'\1 \2', row['comment_text'])
    row['comment_text'] = re.sub(r'([^a-zA-Z0-9\'´` ])([a-zA-Z0-9])', r'\1 \2', row['comment_text'])
    return row


    
        
    
log('Loading data')    
#model = build_model(np.zeros((6000, 300)), 7, [1,1,1,1], 'binary_crossentropy')
model = build_model(np.zeros((6000, 300)), 7, [1,1,1], 'binary_crossentropy')

train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

# Add a no toxic category value so that we can one hot encode this shit
train_df['no_toxic_cat'] = np.where(np.max(train_df[TOXICITY_TYPES].values, axis=1) < 0.5, True, False) 
TOXICITY_TYPES += ['no_toxic_cat']

train_df[IDENTITY_COLUMNS] = train_df[IDENTITY_COLUMNS].fillna(0)

#train_df['no_ident_cat'] = np.where(np.max(train_df[IDENTITY_COLUMNS].values, axis=1) < 0.5, True, False) 
#IDENTITY_COLUMNS += ['no_ident_cat']

log('Separating data')    
train_df = train_df.apply(lambda row: separate_tokens(row), axis=1)
test_df = test_df.apply(lambda row: separate_tokens(row), axis=1)


for column in IDENTITY_COLUMNS + [TARGET_COLUMN] + TOXICITY_TYPES:
    train_df[column] = np.where(train_df[column] >= 0.5, True, False)

log('Adding stats')
train_df = add_comment_stats(train_df)
test_df = add_comment_stats(test_df)

x_train = train_df[TEXT_COLUMN].astype(str)
y_train = train_df['target'].values.astype('float16')
y_aux_train = train_df[TOXICITY_TYPES].values
y_aux2_train = train_df[IDENTITY_COLUMNS].values

x_test = test_df[TEXT_COLUMN].astype(str)

x_train_stats = np.vstack(train_df['stats'].values).astype('float16')
x_test_stats =  np.vstack(test_df['stats'].values).astype('float16')
    

log('Creating tokenizer')
tokenizer = text.Tokenizer(filters=TOKEN_FILTER, lower=False)
tokenizer.fit_on_texts(list(x_train) + list(x_test))


log('Generating comment data')
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)


# Overall
weights = np.ones((len(train_df),)) / 4
# Subgroup
weights += (train_df[IDENTITY_COLUMNS].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (( (train_df['target'].values>=0.5).astype(bool).astype(np.int) +
   (train_df[IDENTITY_COLUMNS].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (( (train_df['target'].values<0.5).astype(bool).astype(np.int) +
   (train_df[IDENTITY_COLUMNS].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
y_train_loss_weight = 1.0 / weights.mean()
y_train_weights = weights


log('Finding identities')
identities = identity_words(train_df)
identities = set([item for sublist in tokenizer.texts_to_sequences(identities) for item in sublist])
contains_identity = np.zeros((x_train.shape[0],), dtype=np.int16)
r = 0
for row in x_train:
    for token in row:
        if token in identities:
            contains_identity[r] = 1
            break
    r += 1
    
contains_identity = contains_identity.astype('bool')
y_aux3_train = contains_identity & y_train.astype('bool')
y_aux3_train_weights = np.ones((len(train_df),))
pos_examples  = np.sum(y_aux3_train)
neg_examples = len(y_aux3_train) - pos_examples
pos_freq = pos_examples/len(y_aux3_train)
neg_freq = neg_examples/len(y_aux3_train)
pos_weight = 1/pos_freq
neg_weight = 1/neg_freq
y_aux3_train_weights[y_aux3_train >= 0.5] *= pos_weight
y_aux3_train_weights[y_aux3_train < 0.5] *= neg_weight
y_aux3_train_loss_weight = 1.0 / y_aux3_train_weights.mean()


if VAL_FRAC > 0:
    log('Separating datasets')
    training_ids = list(range(0, len(x_train)))
    np.random.seed(seed=1337)
    validation_ids = np.random.choice(len(x_train), int(len(x_train)*VAL_FRAC), replace=False)
    validation_ids = sorted(validation_ids)[::-1]
    
    for val_id in validation_ids:
        training_ids.pop(val_id)
        
    df_val = train_df.iloc[validation_ids]
    train_df = train_df.iloc[training_ids]
    
    x_train = x_train[training_ids]
    x_train_stats = x_train_stats[training_ids]
    
    y_train = y_train[training_ids]
    y_aux_train = y_aux_train[training_ids]
    y_aux2_train = y_aux2_train[training_ids]
    y_aux3_train = y_aux3_train[training_ids]
    
    y_train_weights = y_train_weights[training_ids]
    y_aux3_train_weights = y_aux3_train_weights[training_ids]
    
    df_val = convert_dataframe_to_bool(df_val)
    val_text = tokenizer.texts_to_sequences(df_val[TEXT_COLUMN].values)
    val_text = sequence.pad_sequences(val_text, maxlen=MAX_LEN)
    x_val_stats = np.vstack(df_val['stats'].values).astype('float16')
    
test_ids = test_df.id
del test_df
del train_df
log('Building word matrix')
embedding_matrix = build_matrix(tokenizer.word_index, EMBEDDING_FILE)


#losses = [custom_loss, 'categorical_crossentropy', 'categorical_crossentropy', custom_loss]
#loss_weights = [y_train_loss_weight, 1, 1, y_aux3_train_loss_weight]
losses = ['binary_crossentropy', 'categorical_crossentropy', 'binary_crossentropy']
loss_weights = [y_train_loss_weight, 1,  y_aux3_train_loss_weight]
sample_weights = [y_train_weights, np.ones_like(y_train_weights), y_aux3_train_weights]
log('Building models')
#word_model = build_word_model(char_embedding_matrix)

weights = []
checkpoint_predictions = []
val_cp_preds = []

models = []
weights_to_save = ['comment_input','bidirectional','sentence_lstm_relu','sentence_lstm_state_conc','attention_34','dense-block-lstm_dense-0',
 'dense-block-relu-lstm_dense-0','dense-block-lstm_dense-1',
 'dense-block-relu-lstm_dense-1','dense-block-lstm_dense-2','dense-block-relu-lstm_dense-2','dense-block-skip-connection-lstm_dense','dense-block-conc-lstm_dense','dense-block-relulstm_dense',
 'comment_stats','gather_for_pred_conc','summarizer', 'toxicity_dense',
 'toxicity_type_dense','toxic_identity_dense','toxicity','toxicity_type','toxic_identity','concatenate_23']
 
for model_idx in range(NUM_MODELS):
    if model_idx == 1:
        model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weights, losses, l2_reg=0.0001)
    else:
        model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weights, losses)
    for epoch in range(EPOCHS):
        if epoch >= EPOCHS - 2 and TRAIN_LAST:
            model.save_weights('w.h5')
            model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weights, losses, embedding_trainable=True)
            model.load_weights('w.h5')
        model.fit(
            [x_train, x_train_stats], [y_train, y_aux_train, y_aux3_train],
            sample_weight=sample_weights,
            epochs=1,
            batch_size=BATCH_SIZE,
            verbose=2,
            callbacks=[
                LearningRateScheduler(lambda _: 1e-3 * (0.55 ** epoch))
            ]
        )
        if DENSE_ENSAMBLE:
            if epoch != 0:
                model_w_dict = {}
                for layer in model.layers:
                    if layer.name in weights_to_save or 'attention' in layer.name or 'embedding' in layer.name:
                        model_w_dict[layer.name] = layer.get_weights() 
                models.append(model_w_dict)
        else:
            checkpoint_predictions.append(model.predict([x_test, x_test_stats], batch_size=2048)[0].flatten())
            weights.append(2 ** epoch)
            
        if VAL_FRAC > 0:
            preds = model.predict([val_text, x_val_stats], batch_size=2048)[0]
            val_cp_preds.append(preds.flatten())
            check_auc(df_val, preds)

if DENSE_ENSAMBLE:
    log('Creating ensemble')            
    ensemble_model = ensemble(embedding_matrix, models)
    ensemble_model.fit(
        [x_train, x_train_stats], y_train,
        sample_weight=sample_weights[0],
        epochs=1,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[
            LearningRateScheduler(lambda _: 1e-3 * (0.55 ** 1))
        ]
        )
    predictions = ensemble_model.predict([x_test, x_test_stats], batch_size=512).flatten()
else:
    predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
    
submission = pd.DataFrame.from_dict({
    'id': test_ids,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)


if VAL_FRAC > 0:
    check_auc(df_val, np.average(val_cp_preds, weights=weights, axis=0))

#incorrect_df = df_val[np.round(df_val['target']) != np.round(df_val['our_model'])]
def view_incorrect(df, col, false_positives):
    if false_positives:
        return incorrect_df[(incorrect_df[col] >= 0.5) & (incorrect_df['target'] < 0.5)]
    else:
        return incorrect_df[(incorrect_df[col] >= 0.5) & (incorrect_df['target'] >= 0.5)]
#view_incorrect(incorrect_df, 'black', True)

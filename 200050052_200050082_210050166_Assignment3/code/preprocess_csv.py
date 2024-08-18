import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')


def generate_lists_ngram_in_df(file, ngram):
    df = pd.read_csv(file)
    df['tokens'] = df['tokens'].apply(lambda x: x[1:][:-1].split(" "))
    df['ner_tags'] = df['ner_tags'].apply(lambda x: [ 1 if j>0 else 0 for j in [int(i) for i in x.replace('[','').replace(']','').split(" ")]])
    df['pos_tags'] = df['pos_tags'].apply(lambda x: [int(''.join(i.split())) for i in x.replace('[','').replace(']','').split(" ") if ''.join(i.split()).isnumeric()])

    df['next_1_pos_tags'] = df['pos_tags'].apply(lambda x: [47 if i==len(x) else x[i] for i in range(1, len(x)+1)])
    df['prev_1_pos_tags'] = df['pos_tags'].apply(lambda x: [48 if i==-1 else x[i] for i in range(-1, len(x)-1, 1)])

    for i in range(2,ngram+1):
        df[f'next_{str(i)}_pos_tags'] = df[f'next_{str(i-1)}_pos_tags'].apply(lambda x: [47 if i==len(x) else x[i] for i in range(1, len(x)+1)])
    
    for i in range(2,ngram+2):
        df[f'prev_{str(i)}_pos_tags'] = df[f'prev_{str(i-1)}_pos_tags'].apply(lambda x: [48 if i==-1 else x[i] for i in range(-1, len(x)-1, 1)])

    return df

def generate_arrays(df, ngram):
    flat_tokens = [token for sublist in df['tokens'] for token in sublist]
    flat_tokens = [item[1:-1] if item.startswith(("'", '"')) and item.endswith(("'", '"')) else item for item in flat_tokens]
    # 1 : Title, 2 : Upper, 0 : other
    tokenType = [1 if i.istitle() else 2 if i.isupper() else 0 for i in flat_tokens]

    flat_labels = [label for sublist in df['ner_tags'] for label in sublist]
    flat_pos_tags = [label for sublist in df['pos_tags'] for label in sublist]

    flat_next_pos_tags = []
    flat_prev_pos_tags = []

    for i in range(1, ngram+1):
        flat_next_pos_tags.append([label for sublist in df[f'next_{str(i)}_pos_tags'] for label in sublist])
        flat_prev_pos_tags.append([label for sublist in df[f'prev_{str(i)}_pos_tags'] for label in sublist])
        
    # arrays 
    array_ner_tags = np.array(flat_labels)
    array_tokenType = np.array(tokenType)
    array_flat_pos_tags = np.array(flat_pos_tags)
    array_flat_next_pos_tags = np.transpose(np.array(flat_next_pos_tags))
    array_flat_prev_pos_tags = np.transpose(np.array(flat_prev_pos_tags))

    X = np.column_stack((array_tokenType, array_flat_pos_tags, array_flat_next_pos_tags, array_flat_prev_pos_tags))
    y = array_ner_tags

    return X, y


def preprocess_data(train_file, test_file, ngram):
    df_train = generate_lists_ngram_in_df(train_file, ngram)
    df_test = generate_lists_ngram_in_df(test_file, ngram)

    X_train, y_train = generate_arrays(df_train, ngram)
    X_test, y_test = generate_arrays(df_test, ngram)

    return X_train, X_test, y_train, y_test

def generate_features_ner(corpus, ngram):
    words = word_tokenize(corpus)
    tokenType = [1 if i.istitle() else 2 if i.isupper() else 0 for i in words]

    pos_tagset = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11,
                  'DT': 12,
                  'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
                  'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
                  'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
                  'WP': 44, 'WP$': 45, 'WRB': 46}
    
    word_pos_tags = nltk.tag.pos_tag(words)
    num_pos_tags = [pos_tagset[i[1]] for i in word_pos_tags]
    next_pos_tags = []
    prev_pos_tags = []

    next_pos_tags.append([47 if i==len(num_pos_tags) else num_pos_tags[i] for i in range(1, len(num_pos_tags)+1)])
    prev_pos_tags.append([48 if i==-1 else num_pos_tags[i] for i in range(-1, len(num_pos_tags)-1, 1)])

    for j in range(1,ngram):
        next_pos_tags.append([47 if i==len(next_pos_tags[j-1]) else next_pos_tags[j-1][i] for i in range(1, len(next_pos_tags[j-1])+1)])
        prev_pos_tags.append([48 if i==-1 else prev_pos_tags[j-1][i] for i in range(-1, len(prev_pos_tags[j-1])-1, 1)])
    array_tokenType = np.array(tokenType)
    array_pos_tags = np.array(num_pos_tags)
    array_next_pos_tags = np.transpose(np.array(next_pos_tags))
    array_prev_pos_tags = np.transpose(np.array(prev_pos_tags))
    
    X = np.column_stack((array_tokenType, array_pos_tags, array_next_pos_tags, array_prev_pos_tags))
    return X, words

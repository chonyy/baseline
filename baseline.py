import os
import sys
import sklearn_crfsuite
import random
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from tqdm import tqdm


def loadInputFile(path):
    trainingset = list()  # store trainingset [content,content,...]
    # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    position = list()
    mentions = dict()  # store mentions[mention] = Type
    with open(file_path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    datas = file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data = data.split('\n')
        content = data[0]
        trainingset.append(content)
        annotations = data[1:]
        for annot in annotations[1:]:
            # annot= article_id, start_pos, end_pos, entity_text, entity_type
            annot = annot.split('\t')
            position.extend(annot)
            mentions[annot[3]] = annot[4]

    return trainingset, position, mentions


def CRFFormatData(trainingset, position, path):
    if (os.path.isfile(path)):
        os.remove(path)
    outputfile = open(path, 'a', encoding='utf-8')

    # output file lines
    count = 0  # annotation counts in each content
    tagged = list()
    for article_id in range(len(trainingset)):
        trainingset_split = list(trainingset[article_id])
        while '' or ' ' in trainingset_split:
            if '' in trainingset_split:
                trainingset_split.remove('')
            else:
                trainingset_split.remove(' ')
        start_tmp = 0
        for position_idx in range(0, len(position), 5):
            if int(position[position_idx]) == article_id:
                count += 1
                if count == 1:
                    start_pos = int(position[position_idx+1])
                    end_pos = int(position[position_idx+2])
                    entity_type = position[position_idx+4]
                    if start_pos == 0:
                        token = list(
                            trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token_idx == 0:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)

                    else:
                        token = list(trainingset[article_id][0:start_pos])
                        whole_token = trainingset[article_id][0:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue

                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            outputfile.write(output_str)

                        token = list(
                            trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token[0] == '':
                                if token_idx == 1:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type
                            else:
                                if token_idx == 0:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)

                    start_tmp = end_pos
                else:
                    start_pos = int(position[position_idx+1])
                    end_pos = int(position[position_idx+2])
                    entity_type = position[position_idx+4]
                    if start_pos < start_tmp:
                        continue
                    else:
                        token = list(
                            trainingset[article_id][start_tmp:start_pos])
                        whole_token = trainingset[article_id][start_tmp:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            outputfile.write(output_str)

                    token = list(trainingset[article_id][start_pos:end_pos])
                    whole_token = trainingset[article_id][start_pos:end_pos]
                    for token_idx in range(len(token)):
                        if len(token[token_idx].replace(' ', '')) == 0:
                            continue
                        # BIO states
                        if token[0] == '':
                            if token_idx == 1:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type
                        else:
                            if token_idx == 0:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type

                        output_str = token[token_idx] + ' ' + label + '\n'
                        outputfile.write(output_str)
                    start_tmp = end_pos

        token = list(trainingset[article_id][start_tmp:])
        whole_token = trainingset[article_id][start_tmp:]
        for token_idx in range(len(token)):
            if len(token[token_idx].replace(' ', '')) == 0:
                continue

            output_str = token[token_idx] + ' ' + 'O' + '\n'
            outputfile.write(output_str)

        count = 0

        output_str = '\n'
        outputfile.write(output_str)
        ID = trainingset[article_id]

        if article_id % 10 == 0:
            print('Total complete articles:', article_id)

    # close output file
    outputfile.close()


def CRF(x_train, y_train, x_test, y_test):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=10,
        all_possible_transitions=True,
        verbose=False
    )
    crf.fit(x_train, y_train)
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(crf, f)

    y_pred = crf.predict(x_test)
    y_pred_mar = crf.predict_marginals(x_test)

    labels = list(crf.classes_)
    labels.remove('O')
    f1score = metrics.flat_f1_score(
        y_test, y_pred, average='weighted', labels=labels)
    sorted_labels = sorted(labels, key=lambda name: (
        name[1:], name[0]))  # group B and I results
    print(flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3))
    return y_pred, y_pred_mar, f1score


def Dataset(data_path):
    r"""
    load `train.data` and separate into a list of labeled data of each text
    return:
    data_list: a list of lists of tuples, storing tokens and labels (wrapped in tuple) of each text in `train.data`
    traindata_list: a list of lists, storing training data_list splitted from data_list
    testdata_list: a list of lists, storing testing data_list splitted from data_list
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()  # .encode('utf-8').decode('utf-8-sig')
    data_list, data_list_tmp = list(), list()
    article_id_list = list()
    idx = 0
    for row in data:
        data_tuple = tuple()
        if row == '\n':
            article_id_list.append(idx)
            idx += 1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip('\n').split(' ')
            data_tuple = (row[0], row[1])
            data_list_tmp.append(data_tuple)
    if len(data_list_tmp) != 0:
        data_list.append(data_list_tmp)

    # here we random split data into training dataset and testing dataset
    # but you should take `development data` or `test data` as testing data
    # At that time, you could just delete this line,
    # and generate data_list of `train data` and data_list of `development/test data` by this function
    traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = train_test_split(data_list,
                                                                                                          article_id_list,
                                                                                                          test_size=0.33,
                                                                                                          random_state=42)

    return data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list


def Word2Vector(data_list, embedding_dict):
    r"""
    look up word vectors
    turn each word into its pretrained word vector
    return a list of word vectors corresponding to each token in train.data
    """
    embedding_list = list()

    # No Match Word (unknown word) Vector in Embedding
    unk_vector = np.random.rand(*(list(embedding_dict.values())[0].shape))

    for idx_list in range(len(data_list)):
        embedding_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            key = data_list[idx_list][idx_tuple][0]  # token

            if key in embedding_dict:
                value = embedding_dict[key]
            else:
                value = unk_vector
            embedding_list_tmp.append(value)
        embedding_list.append(embedding_list_tmp)
    return embedding_list


def Feature(embed_list):
    r"""
    input features: pretrained word vectors of each token
    return a list of feature dicts, each feature dict corresponding to each token
    """
    feature_list = list()
    for idx_list in range(len(embed_list)):
        feature_list_tmp = list()
        for idx_tuple in range(len(embed_list[idx_list])):
            feature_dict = dict()
            for idx_vec in range(len(embed_list[idx_list][idx_tuple])):
                feature_dict['dim_' + str(idx_vec+1)
                             ] = embed_list[idx_list][idx_tuple][idx_vec]
            feature_list_tmp.append(feature_dict)
        feature_list.append(feature_list_tmp)
    return feature_list


def Preprocess(data_list):
    r"""
    Get the labels of each tokens in train.data.
    Return a list of lists of labels.
    """
    label_list = list()
    for idx_list in range(len(data_list)):
        label_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            label_list_tmp.append(data_list[idx_list][idx_tuple][1])
        label_list.append(label_list_tmp)
    return label_list


def load_word_vector():
    # Load pretrained word vectors.
    # Get a dict of tokens (key) and their pretrained word vectors (value).
    # Pretrained word2vec CBOW word vector: https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305ddf5522015de5479f4701b1
    dim = 0
    word_vecs = {}
    # Open pretrained word vector file.
    with open('cna.cbow.cwe_p.tar_g.512d.0.txt', encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()

            # There 2 integers in the first line: vocabulary_size, word_vector_dim.
            if len(tokens) == 2:
                dim = int(tokens[1])
                continue

            word = tokens[0]
            vec = np.array([float(t) for t in tokens[1:]])
            word_vecs[word] = vec

    print('vocabulary_size: ', len(word_vecs), ' word_vector_dim: ', vec.shape)
    return word_vecs


def FormatOutput(y_pred, testdata_list, testdata_article_id_list):
    r"""
    Format data.
    """
    output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    for test_id, article_pred in enumerate(y_pred):
        pos=0
        start_pos=None
        end_pos=None
        entity_text=None
        entity_type=None
        print(article_pred)
        for pred_id, pred in enumerate(article_pred):
            if pred[0]=='B':
                start_pos=pos
                entity_type=pred[2:]
            elif start_pos is not None and pred[0]=='I' and pred_id == len(article_pred) - 1:
                end_pos=pos
                entity_text=''.join([testdata_list[test_id][position][0] for position in range(start_pos,end_pos+1)])
                line=str(testdata_article_id_list[test_id])+'\t'+str(start_pos)+'\t'+str(end_pos+1)+'\t'+entity_text+'\t'+entity_type
                output+=line+'\n'
            elif start_pos is not None and pred[0]=='I' and article_pred[pred_id+1][0]=='O':
                end_pos=pos
                entity_text=''.join([testdata_list[test_id][position][0] for position in range(start_pos,end_pos+1)])
                line=str(testdata_article_id_list[test_id])+'\t'+str(start_pos)+'\t'+str(end_pos+1)+'\t'+entity_text+'\t'+entity_type
                output+=line+'\n'
            
            pos+=1     
    output_path = 'output.tsv'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output

def dataToList(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()  # .encode('utf-8').decode('utf-8-sig')
    data_list, data_list_tmp = list(), list()
    article_id_list = list()
    idx = 0
    for row in data:
        data_tuple = tuple()
        if row == '\n':
            article_id_list.append(idx)
            idx += 1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip('\n').split(' ')
            data_tuple = (row[0], row[1])
            data_list_tmp.append(data_tuple)
    if len(data_list_tmp) != 0:
        data_list.append(data_list_tmp)

    return data_list, article_id_list

def Word2Vector2(data_list, embedding_dict):
    embedding_list = list()

    # No Match Word (unknown word) Vector in Embedding
    unk_vector=np.random.rand(*(list(embedding_dict.values())[0].shape))

    for idx_list in range(len(data_list)):
        embedding_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            key = data_list[idx_list][idx_tuple][0] # token

            if key in embedding_dict:
                value = embedding_dict[key]
            else:
                value = unk_vector
            embedding_list_tmp.append(value)
        embedding_list.append(embedding_list_tmp)
    return embedding_list

def dataToList(path):
  lines = None
  dataList = []
  with open(path, 'r', encoding='utf8') as f:
    lines = f.readlines()

  for l in lines:
    dataList.append(l[0])

  return dataList

def readToData(path, outputPath):
  lines = None
  with open(path, 'r', encoding='utf8') as f:
    lines = f.readlines()

  toremove = []

  for l in lines:
    if(l.startswith('article') or l.startswith('-----')):
      toremove.append(l)
    if(len(l) == 1):
      toremove.append(l)

  for t in toremove:
    lines.remove(t)
    # print(t)

  with open(outputPath, 'w', encoding='utf8') as f:
    for l in lines[:3]:
      for c in l:
        if(c != '\n'):
          f.write(c + '\n')

def readToData2(path):
  lines = None
  with open(path, 'r', encoding='utf8') as f:
    lines = f.readlines()

  data = []
  data_tmp = []
  for idx, l in enumerate(lines):
    
    if(l == None or len(l) == 1):
      continue
    elif(l.startswith('article')):
      data_tmp = []
    elif(l.startswith('-----')):
      data.append(data_tmp)
    else:
      for c in l:
        data_tmp.append(c)
  return data

if __name__ == '__main__':
    # word_vecs = None
    # with open('word2vec.pkl', 'rb') as f:
    #     word_vecs = pickle.load(f)
    # dataList = readToData2('dev.txt')
    # print(len(dataList))
    # print(len(dataList[0]))
    # print(dataList[0][0])
    # embList = Word2Vector2(dataList, word_vecs)
    # print(embList[0][0])
    # x_test = Feature(embList)
    # with open('model.pkl', 'rb') as f:
    #     crf = pickle.load(f)
    #     y_pred = crf.predict(x_test)
    #     output = FormatOutput(y_pred, dataList, np.arange(len(x_test)))
        # print(output)
    # Set random seed.
    np.random.seed(25)
    random.seed(25)

    # Load data.
    file_path = './train_2.txt'
    trainingset, position, mentions = loadInputFile(file_path)

    # # Format data.
    data_path = 'data/sample3.data'
    CRFFormatData(trainingset, position, data_path)

    # # print(len(trainingset[0]))

    # # test_path = 'dev.txt'
    # # format

    # #Load pretrained word vector.
    # word_vecs = load_word_vector()

    # # Load formated data and split data.
    # data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = Dataset(
    #     data_path)


    # # testDataList, testIDList = dataToList('dev.txt')

    # # print(testDataList[0])

    # # Load Word Embedding
    # trainembed_list = Word2Vector(traindata_list, word_vecs)
    # testembed_list = Word2Vector(testdata_list, word_vecs)

    # # CRF - Train Data (Augmentation Data)
    # x_train = Feature(trainembed_list)
    # y_train = Preprocess(traindata_list)

    # # CRF - Test Data (Golden Standard)
    # x_test = Feature(testembed_list)
    # y_test = Preprocess(testdata_list)

    # # with open('model.pkl', 'rb') as f:
    # #     crf = pickle.load(f)
    # #     y_pred = crf.predict(x_test)
    # #     print(x_test)
    # #     print(y_pred)

    # y_pred, y_pred_mar, f1score = CRF(x_train, y_train, x_test, y_test)

    # print(f1score)

    # output = FormatOutput(y_pred)
    # print(output)

# coding : utf-8

import argparse
import numpy as np
import tensorflow as tf
import scipy
from scipy import stats

MODELNAME = ''
SAVEPATH = ''

def getWordVector(args):

    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if vals[0] == '<unk>':
                continue
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(vectors.keys())

    print vocab_size

    vocab = {}
    ivocab = {}

    for idx, key in enumerate(vectors.keys()):
        vocab[key] = idx
        ivocab[idx] = key

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    # W_norm = np.zeros(W.shape)
    # d = (np.sum(W ** 2, 1) ** (0.5))
    # W_norm = (W.T / d).T

    return W, vocab, ivocab


def initEvalData():
    evalData = {}
    prefix = '../../data/word_similarity_datasets/'
    evalData['scwc'] = prefix + 'scws.txt'
    evalData['mc'] = prefix + 'EN-MC-30.txt'
    evalData['men-3k'] = prefix + 'EN-MEN-TR-3k.txt'
    evalData['MTurk-287'] = prefix + 'EN-MTurk-287.txt'
    evalData['MTurk-771'] = prefix + 'EN-MTurk-771.txt'
    evalData['rg'] = prefix + 'EN-RG-65.txt'
    evalData['ws353-s'] = prefix + 'EN-WS-353-SIM.txt'
    evalData['ws353-r'] = prefix + 'EN-WS-353-REL.txt'
    evalData['ws353-a'] = prefix + 'EN-WS-353-ALL.txt'
    evalData['yp'] = prefix + 'EN-YP-130.txt'
    evalData['simlex999'] = prefix + 'SimLex-999.txt'
    evalData['rw'] = prefix + 'EN-RW-STANFORD.txt'

    return evalData


def evaluate_vectors(W, vocab, ivocab, evalDataPaths):
    global MODELNAME
    global SAVEPATH

    """Evaluate the trained word vectors on a variety of tasks"""

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    f = open(SAVEPATH + MODELNAME + '.txt', 'w')

    for key, filename in evalDataPaths.items():

        full_data = getSimData(filename, key)

        firstWordIdxes, secondWordIdxes, targetScores = Word2Idx(full_data, vocab)

        totalNum = len(full_data)
        containNum = len(firstWordIdxes)

        total_string = '{} dataset: total vob is {} , oov is {} \n'.format(key, totalNum, totalNum - containNum)
        # print total_string
        f.write(total_string)

        simScores = calSimilar(W, firstWordIdxes, secondWordIdxes)

        spr = scipy.stats.spearmanr(simScores, targetScores)

        spr_string = 'Spearman correlation is {} with pvalue {} \n'.format(spr.correlation, spr.pvalue)
        # print spr_string
        f.write(spr_string)

        pear = scipy.stats.pearsonr(simScores, targetScores)
        pear_string = 'Pearson correlation ' + str(pear) + ' \n'
        # print pear_string
        f.write(pear_string)

        f.flush()

    f.close()

def getSimData(filename, key):
    data = []

    if key == 'scwc':
        with open(filename, 'r') as f:
            for line in f:
                spInfo = line.rstrip().split('\t')
                data.append([spInfo[1], spInfo[3], spInfo[-11]])

    elif key == 'simlex999':
        with open(filename, 'r') as f:
            isFirst = True
            for line in f:
                if isFirst:
                    isFirst = False
                    continue

                spInfo = line.rstrip().split()
                full_data = [spInfo[0], spInfo[1], spInfo[3]]
                data.append(full_data)

    else:
        with open(filename, 'r') as f:
            spdef = ' '
            for line in f:
                spInfo = line.rstrip().split(spdef)
                if len(spInfo) == 1:
                    spdef = '\t'
                    spInfo = line.rstrip().split(spdef)

                data.append(spInfo)

    return data

def Word2Idx(full_data, word2Id):

    firstIdxes = []
    secondIdxes = []
    scores = []

    for data in full_data:

        if data[0] in word2Id.keys() and data[1] in word2Id.keys():

            firstIdxes.append(word2Id[data[0]])
            secondIdxes.append(word2Id[data[1]])
            scores.append(float(data[2]))

    return firstIdxes, secondIdxes, scores

def calSimilar(W, firstWordIdxes, secondWordIdxes):

    scores = []

    for first, second in zip(firstWordIdxes, secondWordIdxes):

        scores.append(1. - scipy.spatial.distance.cosine(W[first, :] + 0.000001, W[second, :] + 0.000001))

    return scores


def main():
    global MODELNAME
    global SAVEPATH

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='glove', type=str)
    # parser.add_argument('--vectors_file', default='/home/public/code/wordEmbedding/code/GloVe/vectors.txt', type=str)
    #
    # parser.add_argument('--model_name', default='fasttext', type=str)
    # parser.add_argument('--vectors_file', default='/home/public/code/wordEmbedding/code/fastText/result/text8_result.vec', type=str)

    parser.add_argument('--model_name', default='word2vec', type=str)
    parser.add_argument('--vectors_file', default='/home/public/code/wordEmbedding/code/word2vec/trunk/vector.txt', type=str)

    parser.add_argument('--save_path', default='/home/public/code/wordEmbedding/result/wordSim/', type=str)
    args = parser.parse_args()

    MODELNAME = args.model_name
    SAVEPATH = args.save_path

    W_norm, vocab, ivocab = getWordVector(args)
    evalDataMap = initEvalData()

    evaluate_vectors(W_norm, vocab, ivocab, evalDataMap)

if __name__ == "__main__":

    # evalDataPath = initEvalData()
    #
    # for key, filename in evalDataPath.items():
    #     print('-----------------' + key + '-------------------')
    #     getSimData(filename, key)

    main()

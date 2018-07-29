# coding : utf-8

import argparse
import numpy as np
import tensorflow as tf
import scipy
from scipy import stats

import sys

sys.path.append("/home/public/code/wordEmbedding/code/word2gm")

import word2gm_loader

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), '/home/public/code/wordEmbedding/data/word2vec_ops.so'))


MODELNAME = ''
SAVEPATH = ''

def getModel(args):
    word2mixgauss = word2gm_loader.Word2GM(save_path=args.model_path)
    return word2mixgauss

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


def evaluate_vectors(model, evalDataPaths):
    global MODELNAME
    global SAVEPATH

    """Evaluate the trained word vectors on a variety of tasks"""

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    f = open(SAVEPATH + MODELNAME + '.txt', 'w')

    for key, filename in evalDataPaths.items():

        full_data = getSimData(filename, key)

        firstWordIdxes, secondWordIdxes, targetScores = Word2Idx(full_data, model)

        totalNum = len(full_data)
        containNum = len(firstWordIdxes)

        total_string = '{} dataset: total vob is {} , oov is {} \n'.format(key, totalNum, totalNum - containNum)
        # print total_string
        f.write(total_string)

        simScores = calSimilar(model, firstWordIdxes, secondWordIdxes)

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

def Word2Idx(full_data, model):

    firstIdxes = []
    secondIdxes = []
    scores = []

    for data in full_data:

        if model.words_to_idxs([data[0]]) > 0 and model.words_to_idxs([data[1]]) > 0:

            firstIdxes.append(model.words_to_idxs([data[0]]))
            secondIdxes.append(model.words_to_idxs([data[1]]))
            scores.append(float(data[2]))

    return firstIdxes, secondIdxes, scores

def calSimilar(model, firstWordIdxes, secondWordIdxes):

    scores = []

    for first, second in zip(firstWordIdxes, secondWordIdxes):

        scores.append(model.maxdot(first[0], second[0]))

    return scores


def main():
    global MODELNAME
    global SAVEPATH

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='glove', type=str)
    # parser.add_argument('--vectors_file', default='/home/public/code/wordEmbedding/code/GloVe/vectors.txt', type=str)

    parser.add_argument('--model_name', default='word2gm', type=str)
    parser.add_argument('--model_path', default='/home/public/code/wordEmbedding/code/word2gm/modelfiles/t8-2s-e10-v05-lr05d-mc100-ss5-nwout-adg-win10', type=str)

    parser.add_argument('--save_path', default='/home/public/code/wordEmbedding/result/wordSim/', type=str)
    args = parser.parse_args()

    MODELNAME = args.model_name
    SAVEPATH = args.save_path

    model = getModel(args)
    evalDataMap = initEvalData()

    evaluate_vectors(model, evalDataMap)

if __name__ == "__main__":

    # evalDataPath = initEvalData()
    #
    # for key, filename in evalDataPath.items():
    #     print('-----------------' + key + '-------------------')
    #     getSimData(filename, key)

    sess = tf.Session()
    main()

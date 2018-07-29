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
    prefix = '/home/public/code/wordEmbedding/code/GloVe/eval/question-data/'
    evalData['capital-common-countries'] = prefix + 'capital-common-countries.txt'
    evalData['capital-world'] = prefix + 'capital-world.txt'
    evalData['currency'] = prefix + 'currency.txt'
    evalData['city-in-state'] = prefix + 'city-in-state.txt'
    evalData['family'] = prefix + 'family.txt'
    evalData['gram1-adjective-to-adverb'] = prefix + 'gram1-adjective-to-adverb.txt'
    evalData['gram2-opposite'] = prefix + 'gram2-opposite.txt'
    evalData['gram3-comparative'] = prefix + 'gram3-comparative.txt'
    evalData['gram4-superlative.txt'] = prefix + 'gram4-superlative.txt'
    evalData['gram5-present-participle'] = prefix + 'gram5-present-participle.txt'
    evalData['gram6-nationality-adjective'] = prefix + 'gram6-nationality-adjective.txt'
    evalData['gram7-past-tense'] = prefix + 'gram7-past-tense.txt'
    evalData['gram8-plural'] = prefix + 'gram8-plural.txt'
    evalData['gram9-plural-verbs'] = prefix + 'gram9-plural-verbs.txt'

    return evalData

def evaluate_vectors(model, evalDataMap):
    """Evaluate the trained word vectors on a variety of tasks"""
    global SAVEPATH
    global MODELNAME

    save_file = open(SAVEPATH + MODELNAME + '.txt', 'w')

    print SAVEPATH + MODELNAME + '.txt'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    for key, filepath in evalDataMap.items():
        with open('%s' % (filepath), 'r') as f:

            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [model.words_to_idxs(x) for x in full_data if all(wordId > 0 for wordId in model.words_to_idxs(x))]

        correct = []

        for ind, wordIds in enumerate(data):


            wordIdA = wordIds[0]
            wordIdB = wordIds[1]
            wordIdC = wordIds[2]
            wordIdD = wordIds[3]

            dotResult = model.maxdot(wordIdA, wordIdB)

            minDist = float('inf')
            mostProbWordId = -1

            for wordId in xrange(1, len(model.id2word)):

                curDotResult = model.maxdot(wordIdC, wordId)

                dist = abs(curDotResult - dotResult)

                if dist < minDist:
                    minDist = dist
                    mostProbWordId = wordId

            if wordIdD == mostProbWordId:

                correct.append(1.0)

            else:

                correct.append(0.0)

            # print model.id2word[wordIdA], model.id2word[wordIdB], model.id2word[wordIdC], model.id2word[wordIdD], model.id2word[mostProbWordId]
            # print model.maxdot(wordIdA, wordIdB)
            # print model.maxdot(wordIdC, wordIdD)
            # print model.maxdot(wordIdC, mostProbWordId)
            # print '--------------------------------------------------'

            print 'finish ' + str(ind)


        count_tot = count_tot + len(correct)
        correct_tot = correct_tot + sum(correct)
        if not key.startswith('gram'):
            count_sem = count_sem + len(correct)
            correct_sem = correct_sem + sum(correct)
        else:
            count_syn = count_syn + len(correct)
            correct_syn = correct_syn + sum(correct)

        print("%s:" % filepath)
        print('ACCURACY TOP1: %.2f%% (%d/%d)' %
            (np.mean(correct) * 100, np.sum(correct), len(correct)))

        string = key + '\n'
        save_file.write(string)
        string = 'ACCURACY TOP1: %.2f%% (%d/%d)\n'%(np.mean(correct) * 100, np.sum(correct), len(correct))
        save_file.write(string)

    print('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    print('Semantic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    print('Syntactic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))

    string = 'Questions seen/total: %.2f%% (%d/%d)\n' %(100 * count_tot / float(full_count), count_tot, full_count)
    save_file.write(string)

    string = 'Semantic accuracy: %.2f%%  (%i/%i)\n' %(100 * correct_sem / float(count_sem), correct_sem, count_sem)
    save_file.write(string)

    string = 'Syntactic accuracy: %.2f%%  (%i/%i)\n' %(100 * correct_syn / float(count_syn), correct_syn, count_syn)
    save_file.write(string)

    string = 'Total accuracy: %.2f%%  (%i/%i)\n' %(100 * correct_tot / float(count_tot), correct_tot, count_tot)
    save_file.write(string)

    save_file.flush()
    save_file.close()

def main():
    global MODELNAME
    global SAVEPATH

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='glove', type=str)
    # parser.add_argument('--vectors_file', default='/home/public/code/wordEmbedding/code/GloVe/vectors.txt', type=str)
    #
    # parser.add_argument('--model_name', default='fasttext', type=str)
    # parser.add_argument('--vectors_file', default='/home/public/code/wordEmbedding/code/fastText/result/text8_result.vec', type=str)

    # parser.add_argument('--model_name', default='word2vec', type=str)
    # parser.add_argument('--vectors_file', default='/home/public/code/wordEmbedding/code/word2vec/trunk/vector.txt', type=str)

    parser.add_argument('--model_name', default='word2gm', type=str)
    parser.add_argument('--model_path', default='/home/public/code/wordEmbedding/code/word2gm/modelfiles/t8-2s-e10-v05-lr05d-mc100-ss5-nwout-adg-win10', type=str)

    parser.add_argument('--save_path', default='/home/public/code/wordEmbedding/result/wordEntail/', type=str)
    args = parser.parse_args()

    MODELNAME = args.model_name
    SAVEPATH = args.save_path

    model = getModel(args)
    evalDataMap = initEvalData()

    # for index in xrange(100):
    #     print index, ivocab[index]

    evaluate_vectors(model, evalDataMap)

if __name__ == "__main__":
    main()

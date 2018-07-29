import argparse
import numpy as np
import tensorflow as tf

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
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    return W_norm, vocab, ivocab

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

def evaluate_vectors(W, vocab, ivocab, evalDataMap):
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
            data = [x for x in full_data if all(word in vocab for word in x)]

        # print type(data), type(data[0]), data[0]

        indices = np.array([[vocab[word] for word in row] for row in data])

        ind1, ind2, ind3, ind4 = indices.T

        # print ind1[0], ind2[0], ind3[0], ind4[0]
        # print W[ind1[0]]
        # print W[ind2[0]]
        # print W[ind3[0]]
        # print W[ind4[0]]

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :] + W[ind3[subset], :])

            #cosine similarity if input W has been normalized
            dist = np.dot(W, pred_vec.T)

            for k in range(len(subset)):
                dist[ind1[subset[k]], k] = -np.Inf
                dist[ind2[subset[k]], k] = -np.Inf
                dist[ind3[subset[k]], k] = -np.Inf

            # predicted word index
            predictions[subset] = np.argmax(dist, 0).flatten()

        val = (ind4 == predictions) # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if not key.startswith('gram'):
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

        print("%s:" % filepath)
        print('ACCURACY TOP1: %.2f%% (%d/%d)' %
            (np.mean(val) * 100, np.sum(val), len(val)))

        string = key + '\n'
        save_file.write(string)
        string = 'ACCURACY TOP1: %.2f%% (%d/%d)\n'%(np.mean(val) * 100, np.sum(val), len(val))
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
    parser.add_argument('--model_name', default='glove', type=str)
    parser.add_argument('--vectors_file', default='/home/public/code/wordEmbedding/code/GloVe/vectors.txt', type=str)
    #
    # parser.add_argument('--model_name', default='fasttext', type=str)
    # parser.add_argument('--vectors_file', default='/home/public/code/wordEmbedding/code/fastText/result/text8_result.vec', type=str)

    # parser.add_argument('--model_name', default='word2vec', type=str)
    # parser.add_argument('--vectors_file', default='/home/public/code/wordEmbedding/code/word2vec/trunk/vector.txt', type=str)

    parser.add_argument('--save_path', default='/home/public/code/wordEmbedding/result/wordEntail/', type=str)
    args = parser.parse_args()

    MODELNAME = args.model_name
    SAVEPATH = args.save_path

    W_norm, vocab, ivocab = getWordVector(args)
    evalDataMap = initEvalData()

    # for index in xrange(100):
    #     print index, ivocab[index]

    evaluate_vectors(W_norm, vocab, ivocab, evalDataMap)

if __name__ == "__main__":
    main()


/*
This file is based on or incorporates material from the projects listed below (collectively, "Third Party Code"). 
Microsoft is not the original author of the Third Party Code. The original copyright notice and the license under which Microsoft received such Third Party Code, 
are set forth below. Such licenses and notices are provided for informational purposes only. Microsoft, not the third party, licenses the Third Party Code to you 
under the terms set forth in the EULA for the Microsoft Product. Microsoft reserves all rights not expressly granted under this agreement, whether by implication, 
estoppel or otherwise. 

RNNLM 0.3e by Tomas Mikolov

Provided for Informational Purposes Only

BSD License
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other 
materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include "rnnlmlib.h"

using namespace std;

int argPos(char *str, int argc, char **argv)
{
	int a;

	for (a=1; a<argc; a++) if (!strcmp(str, argv[a])) return a;

	return -1;
}

int main(int argc, char **argv)
{
	int i;

	int debug_mode=1;

	int fileformat=TEXT;

	int train_mode=0;
	int valid_data_set=0;
	int test_data_set=0;
	int rnnlm_file_set=0;
	int fea_file_set=0;
	int fea_valid_file_set=0;
	int fea_matrix_file_set=0;
	double feature_gamma=0.9;

	int alpha_set=0, train_file_set=0;

	int class_size=100;
	int old_classes=0;
	float lambda=0.75;
	float gradient_cutoff=15;
	float dynamic=0;
	float starting_alpha=0.1;
	float regularization=0.0000001;
	float min_improvement=1.003;
	int hidden_size=30;
	int compression_size=0;
	long long direct=0;
	int direct_order=3;
	int bptt=0;
	int bptt_block=10;
	int gen=0;
	int savewp=0;
	int fea_size=0;
	int independent=0;
	int use_lmprob=0;
	int rand_seed=1;
	int nbest=0;
	int one_iter=0;
	int anti_k=0;

	char train_file[MAX_STRING];
	char valid_file[MAX_STRING];
	char test_file[MAX_STRING];
	char rnnlm_file[MAX_STRING];
	char lmprob_file[MAX_STRING];
	char fea_file[MAX_STRING];
	char fea_valid_file[MAX_STRING];
	char fea_matrix_file[MAX_STRING];

	FILE *f;

	if (argc==1) {
		//printf("Help\n");

		printf("Recurrent neural network based language modeling toolkit v 0.3d\n\n");

		printf("Options:\n");

		//
		printf("Parameters for training phase:\n");

		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train rnnlm model\n");

		printf("\t-class <int>\n");
		printf("\t\tWill use specified amount of classes to decompose vocabulary; default is 100\n");

		printf("\t-old-classes\n");
		printf("\t\tThis will use old algorithm to compute classes, which results in slower models but can be a bit more precise\n");

		printf("\t-rnnlm <file>\n");
		printf("\t\tUse <file> to store rnnlm model\n");

		printf("\t-binary\n");
		printf("\t\tRnnlm model will be saved in binary format (default is plain text)\n");

		printf("\t-valid <file>\n");
		printf("\t\tUse <file> as validation data\n");

		printf("\t-alpha <float>\n");
		printf("\t\tSet starting learning rate; default is 0.1\n");

		printf("\t-beta <float>\n");
		printf("\t\tSet L2 regularization parameter; default is 1e-7\n");

		printf("\t-hidden <int>\n");
		printf("\t\tSet size of hidden layer; default is 30\n");

		printf("\t-compression <int>\n");
		printf("\t\tSet size of compression layer; default is 0 (not used)\n");

		printf("\t-direct <int>\n");
		printf("\t\tSets size of the hash for direct connections with n-gram features in millions; default is 0\n");

		printf("\t-direct-order <int>\n");
		printf("\t\tSets the n-gram order for direct connections (max %d); default is 3\n", MAX_NGRAM_ORDER);

		printf("\t-bptt <int>\n");
		printf("\t\tSet amount of steps to propagate error back in time; default is 0 (equal to simple RNN)\n");

		printf("\t-bptt-block <int>\n");
		printf("\t\tSpecifies amount of time steps after which the error is backpropagated through time in block mode (default 10, update at each time step = 1)\n");

		printf("\t-one-iter\n");
		printf("\t\tWill cause training to perform exactly one iteration over training data (useful for adapting final models on different data etc.)\n");

		printf("\t-anti-kasparek <int>\n");
		printf("\t\tModel will be saved during training after processing specified amount of words\n");

		printf("\t-min-improvement <float>\n");
		printf("\t\tSet minimal relative entropy improvement for training convergence; default is 1.003\n");

		printf("\t-gradient-cutoff <float>\n");
		printf("\t\tSet maximal absolute gradient value (to improve training stability, use lower values; default is 15, to turn off use 0)\n");

		//

		printf("Parameters for testing phase:\n");

		printf("\t-rnnlm <file>\n");
		printf("\t\tRead rnnlm model from <file>\n");

		printf("\t-test <file>\n");
		printf("\t\tUse <file> as test data to report perplexity\n");

		printf("\t-lm-prob\n");
		printf("\t\tUse other LM probabilities for linear interpolation with rnnlm model; see examples at the rnnlm webpage\n");

		printf("\t-lambda <float>\n");
		printf("\t\tSet parameter for linear interpolation of rnnlm and other lm; default weight of rnnlm is 0.75\n");

		printf("\t-dynamic <float>\n");
		printf("\t\tSet learning rate for dynamic model updates during testing phase; default is 0 (static model)\n");

		//

		printf("Additional parameters:\n");

		printf("\t-gen <int>\n");
		printf("\t\tGenerate specified amount of words given distribution from current model\n");

		printf("\t-save-word-projections\n");
		printf("\t\tThis will create file 'word_projections.txt' and save the word feature vectors for a given model in this text file.\n");

		printf("\t-independent\n");
		printf("\t\tWill erase history at end of each sentence (if used for training, this switch should be used also for testing & rescoring)\n");

		printf("\t-features <file>\n");
		printf("\t\tWill use additional input features (one row of float numbers, with constant length, specified for every word). Has to be specified for both train and test phases.\n");

		printf("\t-features-valid <file>\n");
		printf("\t\tContains feature vectors for the validation data. Has to be specified for the training phase.\n");

		printf("\t-feature-matrix <file>\n");
		printf("\t\tAlternative to the previous two switches: has to be specified just for training; <file> is supposed to contain on each line word and its feature vector (has to be constant size), only single spaces as delimiters\n");

		printf("\t-feature-gamma <float>\n");
		printf("\t\tExponential decay value for features that get automatically computed from the feature matrix; default 0.9, should be in range <0;1)\n");

		printf("\nExamples:\n");
		printf("rnnlm -train train -rnnlm model -valid valid -hidden 50\n");
		printf("rnnlm -rnnlm model -test test\n");
		printf("\n");

		return 0;	//***
	}


	//set debug mode
	i=argPos((char *)"-debug", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: debug mode not specified!\n");
			return 0;
		}

		debug_mode=atoi(argv[i+1]);

		if (debug_mode>0)
			printf("debug mode: %d\n", debug_mode);
	}


	//search for train file
	i=argPos((char *)"-train", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: training data file not specified!\n");
			return 0;
		}

		strcpy(train_file, argv[i+1]);

		if (debug_mode>0)
			printf("train file: %s\n", train_file);

		f=fopen(train_file, "rb");
		if (f==NULL) {
			printf("ERROR: training data file not found!\n");
			return 0;
		}

		train_mode=1;

		train_file_set=1;
	}


	//set one-iter
	i=argPos((char *)"-one-iter", argc, argv);
	if (i>0) {
		one_iter=1;

		if (debug_mode>0)
			printf("Training for one iteration\n");
	}


	//search for validation file
	i=argPos((char *)"-valid", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: validation data file not specified!\n");
			return 0;
		}

		strcpy(valid_file, argv[i+1]);

		if (debug_mode>0)
			printf("valid file: %s\n", valid_file);

		f=fopen(valid_file, "rb");
		if (f==NULL) {
			printf("ERROR: validation data file not found!\n");
			return 0;
		}

		valid_data_set=1;
	}

	if (train_mode && !valid_data_set) {
		if (one_iter==0) {
			printf("ERROR: validation data file must be specified for training!\n");
			return 0;
		}
	}


	//set nbest rescoring mode
	i=argPos((char *)"-nbest", argc, argv);
	if (i>0) {
		nbest=1;
		if (debug_mode>0)
			printf("Processing test data as list of nbests\n");
	}


	//search for test file
	i=argPos((char *)"-test", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: test data file not specified!\n");
			return 0;
		}

		strcpy(test_file, argv[i+1]);

		if (debug_mode>0)
			printf("test file: %s\n", test_file);


		if (nbest && (!strcmp(test_file, "-"))) ; else {
			f=fopen(test_file, "rb");
			if (f==NULL) {
				printf("ERROR: test data file not found!\n");
				return 0;
			}
		}

		test_data_set=1;
	}


	//search for features file
	i=argPos((char *)"-features", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: features file not specified!\n");
			return 0;
		}

		strcpy(fea_file, argv[i+1]);

		if (debug_mode>0)
			printf("features file: %s\n", fea_file);

		f=fopen(fea_file, "rb");
		if (f==NULL) {
			printf("ERROR: feature file not found!\n");
			return 0;
		} else {
			int a;

			a=0;
			fread(&a, sizeof(a), 1, f);

			if (debug_mode>0)
				printf("feature vector size: %d\n", a);

			fea_size=a;
		}

		fea_file_set=1;
	}


	//search for features file for the valid set
	i=argPos((char *)"-features-valid", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: features for validation file not specified!\n");
			return 0;
		}

		strcpy(fea_valid_file, argv[i+1]);

		if (debug_mode>0)
			printf("features for validation file: %s\n", fea_valid_file);

		f=fopen(fea_file, "rb");
		if (f==NULL) {
			printf("ERROR: validation feature file not found!\n");
			return 0;
		} 

		fea_valid_file_set=1;
	}


	//search for feature matrix
	i=argPos((char *)"-feature-matrix", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: feature matrix file not specified!\n");
			return 0;
		}

		strcpy(fea_matrix_file, argv[i+1]);

		if (debug_mode>0)
			printf("feature matrix file: %s\n", fea_matrix_file);

		f=fopen(fea_matrix_file, "rb");
		if (f==NULL) {
			printf("ERROR: feature matrix file not found!\n");
			return 0;
		} 

		fea_matrix_file_set=1;
	}

	//search for feature gamma
	i=argPos((char *)"-feature-gamma", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: feature gamma parameter not specified!\n");
			return 0;
		}

		feature_gamma=atof(argv[i+1]);

		if (debug_mode>0)
			printf("Feature gamma (exponential decay paramter): %f\n", feature_gamma);
	}


	//set class size parameter
	i=argPos((char *)"-class", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: amount of classes not specified!\n");
			return 0;
		}

		class_size=atoi(argv[i+1]);

		if (debug_mode>0)
			printf("class size: %d\n", class_size);
	}


	//set old class
	i=argPos((char *)"-old-classes", argc, argv);
	if (i>0) {
		old_classes=1;

		if (debug_mode>0)
			printf("Old algorithm for computing classes will be used\n");
	}


	//set lambda
	i=argPos((char *)"-lambda", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: lambda not specified!\n");
			return 0;
		}

		lambda=atof(argv[i+1]);

		if (debug_mode>0)
			printf("Lambda (interpolation coefficient between rnnlm and other lm): %f\n", lambda);
	}


	//set gradient cutoff
	i=argPos((char *)"-gradient-cutoff", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: gradient cutoff not specified!\n");
			return 0;
		}

		gradient_cutoff=atof(argv[i+1]);

		if (debug_mode>0)
			printf("Gradient cutoff: %f\n", gradient_cutoff);
	}


	//set dynamic
	i=argPos((char *)"-dynamic", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: dynamic learning rate not specified!\n");
			return 0;
		}

		dynamic=atof(argv[i+1]);

		if (debug_mode>0)
			printf("Dynamic learning rate: %f\n", dynamic);
	}


	//set gen
	i=argPos((char *)"-gen", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: gen parameter not specified!\n");
			return 0;
		}

		gen=atoi(argv[i+1]);

		if (debug_mode>0)
			printf("Generating # words: %d\n", gen);
	}

	//set savewp
	i=argPos((char *)"-save-word-projections", argc, argv);
	if (i>0) {
		savewp=1;

		if (debug_mode>0)
			printf("Saving word projections...\n");
	}


	//set independent
	i=argPos((char *)"-independent", argc, argv);
	if (i>0) {
		independent=1;

		if (debug_mode>0)
			printf("Sentences will be processed independently...\n");
	}


	//set learning rate
	i=argPos((char *)"-alpha", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: alpha not specified!\n");
			return 0;
		}

		starting_alpha=atof(argv[i+1]);

		if (debug_mode>0)
			printf("Starting learning rate: %f\n", starting_alpha);
		alpha_set=1;
	}


	//set regularization
	i=argPos((char *)"-beta", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: beta not specified!\n");
			return 0;
		}

		regularization=atof(argv[i+1]);

		if (debug_mode>0)
			printf("Regularization: %f\n", regularization);
	}


	//set min improvement
	i=argPos((char *)"-min-improvement", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: minimal improvement value not specified!\n");
			return 0;
		}

		min_improvement=atof(argv[i+1]);

		if (debug_mode>0)
			printf("Min improvement: %f\n", min_improvement);
	}


	//set anti kasparek
	i=argPos((char *)"-anti-kasparek", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: anti-kasparek parameter not set!\n");
			return 0;
		}

		anti_k=atoi(argv[i+1]);

		if ((anti_k!=0) && (anti_k<10000)) anti_k=10000;

		if (debug_mode>0)
			printf("Model will be saved after each # words: %d\n", anti_k);
	}


	//set hidden layer size
	i=argPos((char *)"-hidden", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: hidden layer size not specified!\n");
			return 0;
		}

		hidden_size=atoi(argv[i+1]);

		if (debug_mode>0)
			printf("Hidden layer size: %d\n", hidden_size);
	}


	//set compression layer size
	i=argPos((char *)"-compression", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: compression layer size not specified!\n");
			return 0;
		}

		compression_size=atoi(argv[i+1]);

		if (debug_mode>0)
			printf("Compression layer size: %d\n", compression_size);
	}


	//set direct connections
	i=argPos((char *)"-direct", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: direct connections not specified!\n");
			return 0;
		}

		direct=atoi(argv[i+1]);

		direct*=1000000;
		if (direct<0) direct=0;

		if (debug_mode>0)
			printf("Direct connections: %dM\n", (int)(direct/1000000));
	}


	//set order of direct connections
	i=argPos((char *)"-direct-order", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: direct order not specified!\n");
			return 0;
		}

		direct_order=atoi(argv[i+1]);
		if (direct_order>MAX_NGRAM_ORDER) direct_order=MAX_NGRAM_ORDER;

		if (debug_mode>0)
			printf("Order of direct connections: %d\n", direct_order);
	}


	//set bptt
	i=argPos((char *)"-bptt", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: bptt value not specified!\n");
			return 0;
		}

		bptt=atoi(argv[i+1]);
		bptt++;
		if (bptt<1) bptt=1;

		if (debug_mode>0)
			printf("BPTT: %d\n", bptt-1);
	}


	//set bptt block
	i=argPos((char *)"-bptt-block", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: bptt block value not specified!\n");
			return 0;
		}

		bptt_block=atoi(argv[i+1]);
		if (bptt_block<1) bptt_block=1;

		if (debug_mode>0)
			printf("BPTT block: %d\n", bptt_block);
	}


	//set random seed
	i=argPos((char *)"-rand-seed", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: Random seed variable not specified!\n");
			return 0;
		}

		rand_seed=atoi(argv[i+1]);

		if (debug_mode>0)
			printf("Rand seed: %d\n", rand_seed);
	}


	//use other lm
	i=argPos((char *)"-lm-prob", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: other lm file not specified!\n");
			return 0;
		}

		strcpy(lmprob_file, argv[i+1]);

		if (debug_mode>0)
			printf("other lm probabilities specified in: %s\n", lmprob_file);

		f=fopen(lmprob_file, "rb");
		if (f==NULL) {
			printf("ERROR: other lm file not found!\n");
			return 0;
		}

		use_lmprob=1;
	}


	//search for binary option
	i=argPos((char *)"-binary", argc, argv);
	if (i>0) {
		if (debug_mode>0)
			printf("Model will be saved in binary format\n");

		fileformat=BINARY;
	}


	//search for rnnlm file
	i=argPos((char *)"-rnnlm", argc, argv);
	if (i>0) {
		if (i+1==argc) {
			printf("ERROR: model file not specified!\n");
			return 0;
		}

		strcpy(rnnlm_file, argv[i+1]);

		if (debug_mode>0)
			printf("rnnlm file: %s\n", rnnlm_file);

		f=fopen(rnnlm_file, "rb");

		rnnlm_file_set=1;
	}
	if (train_mode && !rnnlm_file_set) {
		printf("ERROR: rnnlm file must be specified for training!\n");
		return 0;
	}
	if (test_data_set && !rnnlm_file_set) {
		printf("ERROR: rnnlm file must be specified for testing!\n");
		return 0;
	}
	if (!test_data_set && !train_mode && gen==0 && savewp==0) {
		printf("ERROR: training or testing must be specified!\n");
		return 0;
	}
	if ((gen>0) && !rnnlm_file_set) {
		printf("ERROR: rnnlm file must be specified to generate words!\n");
		return 0;
	}
	if ((savewp>0) && !rnnlm_file_set) {
		printf("ERROR: rnnlm file must be specified to save word projections!\n");
		return 0;
	}


	srand(1);

	if (train_mode) {
		CRnnLM model1;

		model1.setTrainFile(train_file);
		model1.setRnnLMFile(rnnlm_file);
		model1.setFileType(fileformat);

		if (fea_file_set==1) {
			model1.setFeaFile(fea_file);
			model1.setFeaSize(fea_size);
			if (fea_valid_file_set==1) {
				model1.setFeaValidFile(fea_valid_file);
			} else if (one_iter==0) {
				printf("ERROR: For training with features, validation feature file must be specified using -features-valid <file>\n");
				exit(1);
			}
		}

		if (fea_matrix_file_set==1) {
			model1.setFeaMatrixFile(fea_matrix_file);
			model1.setFeatureGamma(feature_gamma);
		}

		model1.setOneIter(one_iter);
		if (one_iter==0) model1.setValidFile(valid_file);

		model1.setClassSize(class_size);
		model1.setOldClasses(old_classes);
		model1.setLearningRate(starting_alpha);
		model1.setGradientCutoff(gradient_cutoff);
		model1.setRegularization(regularization);
		model1.setMinImprovement(min_improvement);
		model1.setHiddenLayerSize(hidden_size);
		model1.setCompressionLayerSize(compression_size);
		model1.setDirectSize(direct);
		model1.setDirectOrder(direct_order);
		model1.setBPTT(bptt);
		model1.setBPTTBlock(bptt_block);
		model1.setRandSeed(rand_seed);
		model1.setDebugMode(debug_mode);
		model1.setAntiKasparek(anti_k);
		model1.setIndependent(independent);

		model1.alpha_set=alpha_set;
		model1.train_file_set=train_file_set;

		model1.trainNet();
	}

	if (test_data_set && rnnlm_file_set) {
		CRnnLM model1;

		model1.setLambda(lambda);
		model1.setRegularization(regularization);
		model1.setDynamic(dynamic);
		model1.setTestFile(test_file);
		model1.setRnnLMFile(rnnlm_file);
		if (fea_file_set==1) {
			model1.setFeaFile(fea_file);
			model1.setFeaSize(fea_size);
		}
		model1.setRandSeed(rand_seed);
		model1.useLMProb(use_lmprob);
		if (use_lmprob) model1.setLMProbFile(lmprob_file);
		model1.setDebugMode(debug_mode);

		if (nbest==0) model1.testNet();
		else model1.testNbest();
	}

	if (gen>0) {
		CRnnLM model1;

		model1.setRnnLMFile(rnnlm_file);
		model1.setDebugMode(debug_mode);
		model1.setRandSeed(rand_seed);
		model1.setGen(gen);

		model1.testGen();
	}

	if (savewp>0) {
		CRnnLM model1;

		model1.setRnnLMFile(rnnlm_file);
		model1.setDebugMode(debug_mode);
		model1.setRandSeed(rand_seed);

		model1.saveWordProjections();
	}


	return 0;
}

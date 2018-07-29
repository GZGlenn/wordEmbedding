#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

RESULTDIR=result
DATADIR=data
INPUTDIR=../../data/

./fasttext skipgram -input "${INPUTDIR}"/text8 -output "${RESULTDIR}"/text8 -lr 0.025 -dim 50 \
  -ws 10 -epoch 10 -minCount 5 -neg 25 -loss ns -bucket 2000000 \
  -minn 3 -maxn 6 -thread 4 -t 1e-4 -lrUpdateRate 100

cut -f 1,2 "${DATADIR}"/rw/rw.txt | awk '{print tolower($0)}' | tr '\t' '\n' > "${DATADIR}"/queries.txt

cat "${DATADIR}"/queries.txt | ./fasttext print-word-vectors "${RESULTDIR}"/text8.bin > "${RESULTDIR}"/vectors_text8.txt

python eval.py -m "${RESULTDIR}"/vectors_text8.txt -d "${DATADIR}"/rw/rw.txt

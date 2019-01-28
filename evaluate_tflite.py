#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import csv
import numpy as np
import sys
import wave

from deepspeech import Model
from multiprocessing import JoinableQueue, Process, Queue, cpu_count
from six.moves import zip, range
from util.text import wer_cer_batch, levenshtein


r'''
This module should e self-contained:
  - build a libdeepspeech.so with TFLite:
    - add a dep in native_client/BUILD against TFlite: '//tensorflow:linux_x86_64': [ "//tensorflow/contrib/lite/kernels:builtin_ops" ]
    - bazel build [...] --copt=-DUSE_TFLITE [...] //native_client:libdeepspeech.so
  - make -C native_client/python/ TFDIR=... bindings
  - setup a virtualenv
  - pip install numpy
  - pip install native_client/python/dist/deepspeech*.whl

Then run with a TF Lite model, alphabet, LM/trie and a CSV test file
'''

BEAM_WIDTH = 500
LM_ALPHA = 0.75
LM_BETA = 1.85
N_FEATURES = 26
N_CONTEXT = 9

def tflite_worker(model, alphabet, lm, trie, queue_in, queue_out):
    ds = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
    ds.enableDecoderWithLM(alphabet, lm, trie, LM_ALPHA, LM_BETA)

    while True:
        msg = queue_in.get()

        fin = wave.open(msg['filename'], 'rb')
        fs = fin.getframerate()
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
        audio_length = fin.getnframes() * (1/16000)
        fin.close()
    
        decoded = ds.stt(audio, fs)
        
        queue_out.put({'prediction': decoded, 'ground_truth': msg['transcript']})
        queue_in.task_done()

def main():
    parser = argparse.ArgumentParser(description='Computing TFLite accuracy')
    parser.add_argument('--model', required=True,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--alphabet', required=True,
                        help='Path to the configuration file specifying the alphabet used by the network')
    parser.add_argument('--lm', required=True,
                        help='Path to the language model binary file')
    parser.add_argument('--trie', required=True,
                        help='Path to the language model trie file created with native_client/generate_trie')
    parser.add_argument('--csv', required=True,
                        help='Path to the CSV source file')
    args = parser.parse_args()

    work_todo = JoinableQueue()   # this is where we are going to store input data
    work_done = Queue()  # this where we are gonna push them out

    processes = []
    for i in range(cpu_count()):
        worker_process = Process(target=tflite_worker, args=(args.model, args.alphabet, args.lm, args.trie, work_todo, work_done), daemon=True, name='tflite_process_{}'.format(i))
        worker_process.start()        # Launch reader() as a separate python process
        processes.append(worker_process)

    print([x.name for x in processes])

    ground_truths = []
    predictions = []

    with open(args.csv, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            work_todo.put({'filename': row['wav_filename'], 'transcript': row['transcript']})
    work_todo.join()

    while (not work_done.empty()):
        msg = work_done.get()
        ground_truths.extend([ msg['ground_truth'] ])
        predictions.extend([ msg['prediction'] ])

    wer, cer = wer_cer_batch(ground_truths, predictions)
    print('Test - WER: %f, CER: %f' % (wer, cer))

if __name__ == '__main__':
    main()

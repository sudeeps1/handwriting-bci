"""
run_step5.py  –  RNN Inference (Step 5)

Runs the trained RNN on each held-out session and saves the outputs.
Then evaluates character/word error rates.

Usage:
    /Users/sudeeps1/anaconda3/envs/bci_tf2/bin/python \
        /Users/sudeeps1/work/bci-project/handwritingBCI/run_step5.py
"""

import os
import sys
import gc
import numpy as np
import scipy.io

repoDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repoDir)

import tensorflow as tf
from charSeqRNN import charSeqRNN, getDefaultRNNArgs
from characterDefinitions import getHandwritingCharacterDefinitions
from rnnEval import evaluateRNNOutput

# ── configuration ────────────────────────────────────────────────────────────
rootDir = os.path.expanduser("~") + "/handwritingBCIData/"

dataDirs = [
    "t5.2019.05.08", "t5.2019.11.25", "t5.2019.12.09", "t5.2019.12.11",
    "t5.2019.12.18", "t5.2019.12.20", "t5.2020.01.06", "t5.2020.01.08",
    "t5.2020.01.13", "t5.2020.01.15",
]

cvPart = "HeldOutTrials"
rnnOutputDir = cvPart

inferenceSaveDir = rootDir + "RNNTrainingSteps/Step5_RNNInference/" + rnnOutputDir
os.makedirs(inferenceSaveDir, exist_ok=True)

# ── Part 1: Run inference for each session ───────────────────────────────────
print("=" * 60)
print("STEP 5 — RNN INFERENCE")
print("=" * 60)

for x in range(len(dataDirs)):
    print(f"\nProcessing dataset {dataDirs[x]}  ({x+1}/{len(dataDirs)})")

    args = getDefaultRNNArgs()
    args['outputDir'] = rootDir + 'RNNTrainingSteps/Step4_RNNTraining/' + rnnOutputDir
    args['loadDir'] = args['outputDir']
    args['mode'] = 'infer'
    args['timeSteps'] = 7500   # long enough for the longest sentence
    args['batchSize'] = 2      # small batch for memory safety
    args['synthBatchSize'] = 0  # no synthetic data during inference

    # Point at one session at a time
    args['sentencesFile_0']      = rootDir + 'Datasets/' + dataDirs[x] + '/sentences.mat'
    args['singleLettersFile_0']  = rootDir + 'Datasets/' + dataDirs[x] + '/singleLetters.mat'
    args['labelsFile_0']         = (rootDir + 'RNNTrainingSteps/Step2_HMMLabels/'
                                    + cvPart + '/' + dataDirs[x] + '_timeSeriesLabels.mat')
    args['syntheticDatasetDir_0'] = (rootDir + 'RNNTrainingSteps/Step3_SyntheticSentences/'
                                     + cvPart + '/' + dataDirs[x] + '_syntheticSentences/')
    args['cvPartitionFile_0']    = rootDir + 'RNNTrainingSteps/trainTestPartitions_' + cvPart + '.mat'
    args['sessionName_0']        = dataDirs[x]

    args['inferenceOutputFileName'] = inferenceSaveDir + '/' + dataDirs[x] + '_inferenceOutputs.mat'
    args['inferenceInputLayer'] = x

    # Keep all 10 input layers so checkpoint restores correctly.
    # In inference mode, charSeqRNN uses args['inferenceInputLayer']
    # directly, so we don't need to mutate dayToLayerMap[0].
    layerMap = list(range(10))
    args['dayToLayerMap']  = str(layerMap)
    args['dayProbability'] = '[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]'

    rnnModel = charSeqRNN(args=args)
    rnnModel.inference()

    # Free memory before loading the next session
    del rnnModel
    gc.collect()

# ── Part 2: Evaluate outputs ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

charDef = getHandwritingCharacterDefinitions()
allErrCounts = []

for x in range(len(dataDirs)):
    print(f'\n-- {dataDirs[x]} --')

    outputs     = scipy.io.loadmat(inferenceSaveDir + '/' + dataDirs[x] + '_inferenceOutputs.mat')
    sentenceDat = scipy.io.loadmat(rootDir + 'Datasets/' + dataDirs[x] + '/sentences.mat')

    errCounts, decSentences = evaluateRNNOutput(
        outputs['outputs'],
        sentenceDat['numTimeBinsPerSentence'] / 2 + 50,
        sentenceDat['sentencePrompt'],
        charDef,
        charStartThresh=0.3,
        charStartDelay=15)

    saveDict = {}
    saveDict['decSentences'] = decSentences
    saveDict['trueSentences'] = sentenceDat['sentencePrompt']
    saveDict.update(errCounts)
    scipy.io.savemat(inferenceSaveDir + '/' + dataDirs[x] + '_errCounts.mat', saveDict)

    cvPartFile = scipy.io.loadmat(rootDir + 'RNNTrainingSteps/trainTestPartitions_' + cvPart + '.mat')
    valIdx = cvPartFile[dataDirs[x] + '_test']

    if len(valIdx) == 0:
        print('No validation sentences for this session.')
        continue

    valAcc = 100 * (1 - np.sum(errCounts['charErrors'][valIdx]) / np.sum(errCounts['charCounts'][valIdx]))
    print(f'Character error rate for this session: {100 - valAcc:.2f}%')
    print('Decoded sentences:')

    for v in np.squeeze(valIdx):
        trueText = sentenceDat['sentencePrompt'][v, 0][0]
        trueText = trueText.replace('>', ' ')
        trueText = trueText.replace('~', '.')
        trueText = trueText.replace('#', '')
        print(f'  True:    {trueText}')
        print(f'  Decoded: {decSentences[v]}')
        print()

    allErrCounts.append(np.stack([
        errCounts['charCounts'][valIdx],
        errCounts['charErrors'][valIdx],
        errCounts['wordCounts'][valIdx],
        errCounts['wordErrors'][valIdx]], axis=0).T)

# ── Part 3: Overall results ─────────────────────────────────────────────────
concatErrCounts = np.squeeze(np.concatenate(allErrCounts, axis=0))
cer = 100 * (np.sum(concatErrCounts[:, 1]) / np.sum(concatErrCounts[:, 0]))
wer = 100 * (np.sum(concatErrCounts[:, 3]) / np.sum(concatErrCounts[:, 2]))

print("\n" + "=" * 60)
print(f"OVERALL CHARACTER ERROR RATE: {cer:.2f}%")
print(f"OVERALL WORD ERROR RATE:      {wer:.2f}%")
print("=" * 60)
print("\nStep 5 complete.")

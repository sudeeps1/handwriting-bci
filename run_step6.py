"""
run_step6.py  –  Apply Bigram Language Model (Step 6)

Requires Kaldi and kaldi-decoders installed in the home directory.
Applies the n-gram language model to the RNN probability matrices.
"""

import os
import sys
import time
import multiprocessing
import numpy as np
import scipy.io
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

repoDir = os.path.dirname(os.path.abspath(__file__)) + '/'
sys.path.insert(0, repoDir)
os.chdir(repoDir)

from parallelBash import parallelBash
from kaldiReadWrite import readKaldiLatticeFile
from rnnEval import wer

# ── configuration ────────────────────────────────────────────────────────────
rootDir = os.path.expanduser("~") + "/handwritingBCIData/"
homeDir = os.path.expanduser("~/")

dataDirs = [
    "t5.2019.05.08", "t5.2019.11.25", "t5.2019.12.09", "t5.2019.12.11",
    "t5.2019.12.18", "t5.2019.12.20", "t5.2020.01.06", "t5.2020.01.08",
    "t5.2020.01.13", "t5.2020.01.15",
]

cvPart = "HeldOutTrials"

import glob

# Add kaldi and kaldi-decoders binaries to PATH
# Standard installation paths:
os.environ['PATH'] += ':'+homeDir+'kaldi-decoders/bin'
os.environ['PATH'] += ':'+homeDir+'kaldi/src/bin'
os.environ['PATH'] += ':'+homeDir+'kaldi/src/lm'
os.environ['PATH'] += ':'+homeDir+'kaldi/src/lmbin'
os.environ['PATH'] += ':'+homeDir+'kaldi/src/fstbin'
os.environ['PATH'] += ':'+homeDir+'kaldi/src/featbin'
os.environ['PATH'] += ':'+homeDir+'kaldi/src/gmmbin'
os.environ['PATH'] += ':'+homeDir+'kaldi/tools/openfst-1.6.7/bin'
os.environ['PATH'] += ':'+homeDir+'kaldi/src/latbin'

# Anaconda fallback search for Kaldi binaries (e.g. lattice-to-nbest)
conda_kaldi_bins = glob.glob(os.path.expanduser("~/anaconda3/pkgs/kaldi-*/bin"))
for path in conda_kaldi_bins:
    os.environ['PATH'] += ':' + path

os.system('chmod +x ' + repoDir+'kaldiLMScripts/bigramLmDecode.sh')
os.system('chmod +x ' + repoDir+'kaldiLMScripts/parseOptions.inc.sh')
os.system('chmod +x ' + repoDir+'kaldiLMScripts/remove_transcript_dummy_boundaries.sh')
os.system('chmod +x ' + repoDir+'kaldiLMScripts/int2sym.pl')

print("=" * 60)
print("STEP 6 — APPLY BIGRAM LANGUAGE MODEL")
print("=" * 60)

# ── Part 1: decode with Kaldi ───────────────────────────────────────────────
for x_data, dataDir in enumerate(dataDirs):
    print(f"\nDecoding {dataDir} ({x_data+1}/{len(dataDirs)})")
    langModelDir = rootDir + 'BigramLM'
    matsDir = rootDir + 'RNNTrainingSteps/Step6_ApplyBigramLM/' + cvPart + '/KaldiMatrices/' + dataDir
    outDir = rootDir + 'RNNTrainingSteps/Step6_ApplyBigramLM/' + cvPart + '/KaldiOutput/' + dataDir
    
    os.makedirs(outDir, exist_ok=True)
    os.system('rm -f ' + outDir + '/*')
    
    scriptFile = repoDir + 'kaldiLMScripts/bigramLmDecode.sh'
    bashScratchDir = rootDir + 'RNNTrainingSteps/bashScratch'
    os.makedirs(bashScratchDir, exist_ok=True)
    bashFilePrefix = bashScratchDir + '/lmDecode'
    
    txtFiles = [f for f in os.listdir(matsDir) if f.endswith('.txt')]
    nFiles = len(txtFiles)    
    nParallelProcesses = max(1, int(multiprocessing.cpu_count() / 2))
    sentenceIdx = np.arange(0, nFiles).astype(np.int32)
    
    argList = []
    for x in range(len(sentenceIdx)):
        newArgs = {
            'acoustic_scale': 1.0,
            'beam': 65,
            'max_active': 5000,
            '1_mainArg': langModelDir,
            '2_mainArg': matsDir + '/kaldiMat_'+str(sentenceIdx[x])+'.txt',
            '3_mainArg': outDir + '/' + str(sentenceIdx[x]) + '_'
        }
        argList.append(newArgs)
        
    parallelBash(argList, scriptFile, bashFilePrefix, nParallelProcesses)
    
    os.system('chmod +x ' + bashFilePrefix+'_master.sh')
    for x in range(nParallelProcesses):
        os.system('chmod +x ' + bashFilePrefix+'_'+str(x)+'.sh')
        
    os.system(bashFilePrefix+'_master.sh')
    
    # Wait until all parallel tasks write their 9 output files per sentence
    expectedFiles = nFiles * 9
    numFilesInDir = 0
    t_start = time.time()
    while numFilesInDir < expectedFiles:
        numFilesInDir = len(os.listdir(outDir))
        sys.stdout.write(f"\rWaiting for outputs... ({numFilesInDir}/{expectedFiles})")
        sys.stdout.flush()
        if time.time() - t_start > 300: # 5 min timeout
            print("\nTimeout waiting for Kaldi outputs.")
            break
        time.sleep(1)
    print("\nDecoding completed.")

# ── Part 2: evaluate performance ────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION RESULTS (Kaldi Bigram Rescoring)")
print("=" * 60)

valErrCounts = []
for dataDir in dataDirs:
    print(f' \n-- {dataDir} -- ')
    sentenceDat = scipy.io.loadmat(rootDir+'Datasets/'+dataDir+'/sentences.mat')
    cvPartFile = scipy.io.loadmat(rootDir+'RNNTrainingSteps/trainTestPartitions_'+cvPart+'.mat')
    valIdx = cvPartFile[dataDir+'_test']
    
    kaldiDir = rootDir+'RNNTrainingSteps/Step6_ApplyBigramLM/'+cvPart+'/KaldiOutput/'+dataDir 
    nFiles = int(len(os.listdir(kaldiDir))/9)   
    
    allErrCounts = []
    decSentences = []
    
    for fileIdx in range(nFiles):
        nbestFile = kaldiDir+'/'+str(fileIdx)+'_transcript.txt'
        acFile = kaldiDir+'/'+str(fileIdx)+'_best_acscore.ark'
        lmFile = kaldiDir+'/'+str(fileIdx)+'_best_lmscore.ark'

        if not os.path.exists(nbestFile):
            continue

        nums, content = readKaldiLatticeFile(nbestFile, 'string')
        _, acScore = readKaldiLatticeFile(acFile, 'numeric')
        _, lmScore = readKaldiLatticeFile(lmFile, 'numeric')

        # N-best list rescoring based on acScore + 2.0 * lmScore
        bestIdx = np.argmin(acScore + 2.0*lmScore)
        decSent = content[bestIdx]
        decSentences.append(decSent)
        
        trueText = sentenceDat['sentencePrompt'][fileIdx,0][0]
        trueText = trueText.replace('>',' ').replace('~','.').replace('#','')
        
        charErrs = wer(list(trueText), list(decSent))
        wordErrs = wer(trueText.split(), decSent.split())
        allErrCounts.append(np.array([charErrs, len(trueText), wordErrs, len(trueText.split())]))

        if fileIdx in valIdx:
            print(f'#{fileIdx}')
            print('True:    ' + trueText)
            print('Decoded: ' + decSent)
            print('')
            valErrCounts.append(np.array([charErrs, len(trueText), wordErrs, len(trueText.split())]))

    if not allErrCounts: continue

    concatCounts = np.stack(allErrCounts, axis=0)
    saveDict = {
        'decSentences': decSentences,
        'trueSentences': sentenceDat['sentencePrompt'],
        'charCounts': concatCounts[:,1],
        'charErrors': concatCounts[:,0],
        'wordCounts': concatCounts[:,3],
        'wordErrors': concatCounts[:,2]
    }
    os.makedirs(rootDir + 'RNNTrainingSteps/Step6_ApplyBigramLM/' + cvPart, exist_ok=True)
    scipy.io.savemat(rootDir + 'RNNTrainingSteps/Step6_ApplyBigramLM/' + cvPart + '/' + dataDir + '_errCounts.mat', saveDict)

# ── Summary ──────────────────────────────────────────────────────────────────
concatErrCounts = np.squeeze(np.stack(valErrCounts, axis=0))
cer = 100 * (np.sum(concatErrCounts[:,0]) / np.sum(concatErrCounts[:,1]))
wer = 100 * (np.sum(concatErrCounts[:,2]) / np.sum(concatErrCounts[:,3]))

print("=" * 60)
print(f'OVERALL CHARACTER ERROR RATE (BIGRAM LM): {cer:.2f}%')
print(f'OVERALL WORD ERROR RATE (BIGRAM LM):      {wer:.2f}%')
print("=" * 60)

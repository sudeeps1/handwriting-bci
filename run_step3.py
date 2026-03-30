"""
run_step3.py  –  Standalone script equivalent to Step3_syntheticDataAugmentation.ipynb
Run with:
    /Users/sudeeps1/anaconda3/envs/bci_tf2/bin/python run_step3.py

On a 16 GB M4 Mac, nParallelProcesses=4 is safe (~8 GB peak).
"""

import os
import sys
import datetime
import multiprocessing

import scipy.io
import tensorflow as tf

# suppress noisy TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# point at the repo so local imports work
repoDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repoDir)

from characterDefinitions import getHandwritingCharacterDefinitions
from makeSyntheticSentences import generateCharacterSequences, extractCharacterSnippets, addSingleLetterSnippets
from dataPreprocessing import normalizeSentenceDataCube

# ── Configuration ──────────────────────────────────────────────────────────────
rootDir = os.path.expanduser("~") + "/handwritingBCIData/"

dataDirs = [
    "t5.2019.05.08", "t5.2019.11.25", "t5.2019.12.09", "t5.2019.12.11",
    "t5.2019.12.18", "t5.2019.12.20", "t5.2020.01.06", "t5.2020.01.08",
    "t5.2020.01.13", "t5.2020.01.15",
]

cvParts = ["HeldOutBlocks", "HeldOutTrials"]

# ⚠️  Reduce if you run out of memory (4 is safe on 16 GB, 8 is usually fine too)
nParallelProcesses = 4

charDef = getHandwritingCharacterDefinitions()
step3Dir = rootDir + "RNNTrainingSteps/Step3_SyntheticSentences"
os.makedirs(step3Dir, exist_ok=True)


def phase1():
    """Build character-snippet libraries for all sessions × CV partitions."""
    print("=" * 60)
    print("Phase 1: building character-snippet libraries")
    print("=" * 60)

    for dataDir in dataDirs:
        print(f"\nProcessing {dataDir}")
        for cvPart in cvParts:
            print(f"  -- {cvPart}", flush=True)

            snippetFile = f"{step3Dir}/{cvPart}/{dataDir}_snippets.mat"
            if os.path.isfile(snippetFile):
                print("     snippet file already exists, skipping.")
                continue

            sentenceDat     = scipy.io.loadmat(f"{rootDir}Datasets/{dataDir}/sentences.mat")
            singleLetterDat = scipy.io.loadmat(f"{rootDir}Datasets/{dataDir}/singleLetters.mat")
            twCubes         = scipy.io.loadmat(
                f"{rootDir}RNNTrainingSteps/Step1_TimeWarping/{dataDir}_warpedCubes.mat"
            )
            cvPartFile      = scipy.io.loadmat(
                f"{rootDir}RNNTrainingSteps/trainTestPartitions_{cvPart}.mat"
            )
            trainPartitionIdx = cvPartFile[f"{dataDir}_train"]

            for x in range(sentenceDat["sentencePrompt"].shape[0]):
                sentenceDat["sentencePrompt"][x, 0][0] = (
                    sentenceDat["sentencePrompt"][x, 0][0].replace("#", "")
                )

            neuralCube = normalizeSentenceDataCube(sentenceDat, singleLetterDat)
            labels = scipy.io.loadmat(
                f"{rootDir}RNNTrainingSteps/Step2_HMMLabels/{cvPart}/{dataDir}_timeSeriesLabels.mat"
            )

            snippetDict = extractCharacterSnippets(
                labels["letterStarts"],
                labels["blankWindows"],
                neuralCube,
                sentenceDat["sentencePrompt"][:, 0],
                sentenceDat["numTimeBinsPerSentence"][:, 0],
                trainPartitionIdx,
                charDef,
            )
            snippetDict = addSingleLetterSnippets(snippetDict, singleLetterDat, twCubes, charDef)

            os.makedirs(f"{step3Dir}/{cvPart}", exist_ok=True)
            scipy.io.savemat(snippetFile, snippetDict)
            print(f"     saved → {snippetFile}")


def phase2():
    """Generate synthetic .tfrecord sentence files using parallel workers."""
    print("\n" + "=" * 60)
    print("Phase 2: generating synthetic sentence .tfrecord files")
    print(f"         (using {nParallelProcesses} parallel workers)")
    print("=" * 60)

    for dataDir in dataDirs:
        print(f"\nProcessing {dataDir}")
        for cvPart in cvParts:
            print(f"  -- {cvPart}", flush=True)

            outputDir = f"{step3Dir}/{cvPart}/{dataDir}_syntheticSentences"
            os.makedirs(outputDir, exist_ok=True)
            os.makedirs(f"{rootDir}bashScratch", exist_ok=True)

            base_args = {
                "nSentences":         256,
                "nSteps":             2400,
                "binSize":            2,
                "wordListFile":       f"{repoDir}/wordList/google-10000-english-usa.txt",
                "rareWordFile":       f"{repoDir}/wordList/rareWordIdx.mat",
                "snippetFile":        f"{step3Dir}/{cvPart}/{dataDir}_snippets.mat",
                "accountForPenState": 1,
                "charDef":            charDef,
                "seed":               datetime.datetime.now().microsecond,
            }

            argList = []
            for x in range(20):
                a = base_args.copy()
                a["saveFile"] = f"{outputDir}/bat_{x}.tfrecord"
                a["seed"] += x
                argList.append(a)

            argList = [a for a in argList if not os.path.isfile(a["saveFile"])]
            if not argList:
                print("     all 20 batches already exist, skipping.")
                continue

            print(f"     generating {len(argList)} missing batch(es)…", flush=True)
            with multiprocessing.Pool(nParallelProcesses) as pool:
                pool.map(generateCharacterSequences, argList)
            print(f"     done → {outputDir}")


if __name__ == "__main__":
    # 'fork' lets workers inherit the already-imported state without re-running
    # the module; avoids the macOS 'spawn' RuntimeError.
    multiprocessing.set_start_method("fork")

    phase1()
    phase2()

    print("\n✅  Step 3 complete.")

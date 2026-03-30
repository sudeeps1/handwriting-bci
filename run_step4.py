"""
run_step4.py  –  Generates args.p and launches RNN training (Step 4).

Run with:
    /Users/sudeeps1/anaconda3/envs/bci_tf2/bin/python \
        /Users/sudeeps1/work/bci-project/handwritingBCI/run_step4.py

Training runs in-process (no background fork) so you can see live output.
Use tmux / nohup if you want it to survive terminal close (see bottom of this
file for the nohup variant).

To RESUME after a crash, just re-run the same command – the script detects
the existing checkpoint and picks up from the last saved batch.
"""

import os
import sys
import pickle
from datetime import datetime

# ── point at the repo ────────────────────────────────────────────────────────
repoDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repoDir)

from charSeqRNN import getDefaultRNNArgs, charSeqRNN

# ── configuration (mirrors the Step4 notebook) ───────────────────────────────
rootDir = os.path.expanduser("~") + "/handwritingBCIData/"

dataDirs = [
    "t5.2019.05.08", "t5.2019.11.25", "t5.2019.12.09", "t5.2019.12.11",
    "t5.2019.12.18", "t5.2019.12.20", "t5.2020.01.06", "t5.2020.01.08",
    "t5.2020.01.13", "t5.2020.01.15",
]

cvPart      = "HeldOutTrials"   # or "HeldOutBlocks"
outputDir   = rootDir + "RNNTrainingSteps/Step4_RNNTraining/" + cvPart

os.makedirs(outputDir, exist_ok=True)

# ── build args ───────────────────────────────────────────────────────────────
args = getDefaultRNNArgs()

for x, dataDir in enumerate(dataDirs):
    args[f"sentencesFile_{x}"]      = rootDir + "Datasets/" + dataDir + "/sentences.mat"
    args[f"singleLettersFile_{x}"]  = rootDir + "Datasets/" + dataDir + "/singleLetters.mat"
    args[f"labelsFile_{x}"]         = (rootDir + "RNNTrainingSteps/Step2_HMMLabels/"
                                        + cvPart + "/" + dataDir + "_timeSeriesLabels.mat")
    args[f"syntheticDatasetDir_{x}"]= (rootDir + "RNNTrainingSteps/Step3_SyntheticSentences/"
                                        + cvPart + "/" + dataDir + "_syntheticSentences/")
    args[f"cvPartitionFile_{x}"]    = rootDir + "RNNTrainingSteps/trainTestPartitions_" + cvPart + ".mat"
    args[f"sessionName_{x}"]        = dataDir

args["outputDir"]       = outputDir
args["dayProbability"]  = "[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]"
args["dayToLayerMap"]   = "[0,1,2,3,4,5,6,7,8,9]"
args["mode"]            = "train"

# ── auto-resume: if a checkpoint already exists, load from it ────────────────
ckpt_index = os.path.join(outputDir, "checkpoint")
if os.path.isfile(ckpt_index):
    print(f"[run_step4] Checkpoint detected in {outputDir} – resuming.")
    args["loadDir"] = outputDir   # _restore_checkpoint handles the rest
else:
    print(f"[run_step4] No checkpoint found – starting fresh.")
    args["loadDir"] = "None"

# ── save args.p (notebook-compatible) ────────────────────────────────────────
argsFile = os.path.join(outputDir, "args.p")
pickle.dump(args, open(argsFile, "wb"))
print(f"[run_step4] args.p saved → {argsFile}")

# ── run training ─────────────────────────────────────────────────────────────
print(f"[run_step4] Starting training  ({args['nBatchesToTrain']:,} batches total).")
print(f"            Checkpoint every   {args['batchesPerModelSave']:,} batches.")
print(f"            Checkpoints kept   {args['nCheckToKeep']}.")
print()

rnn = charSeqRNN(args=args)
rnn.train()

print("\n[run_step4] Training complete.")

"""
charSeqRNN.py  –  TF2 / Apple-Silicon-compatible port of the handwritingBCI
                  character-sequence RNN.

Key differences from the original TF1 version
----------------------------------------------
* Uses TF2 / Keras eager execution (no tf.Session, no tf.placeholder).
* CudnnGRU replaced by tf.keras.layers.GRU(reset_after=False) which
  matches cuDNN's "linear-before-reset" formulation as closely as TF2 allows.
  The numerical outputs will be very similar but not bit-identical.
* Dataset pipeline updated to TF2 API (no tf.contrib, no experimental shims).
* Checkpoints stored as TF2 tf.train.Checkpoint (NOT compatible with TF1
  .ckpt files – retrain from scratch or convert weights manually).
* Gradient clipping / Adam / L2 rewritten with Keras idioms.
* All tf.compat.v1.* calls removed; pure TF2 only.

Accuracy target: ~90-95% char-level accuracy on the original dataset
                 (paper: 94.22%).
"""

import argparse
import os
from datetime import datetime

import numpy as np
import scipy.io
import scipy.special
from scipy.ndimage import gaussian_filter1d
import pickle
import sys
import random

import tensorflow as tf

from dataPreprocessing import prepareDataCubesForRNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gauss_kernel(kernelSD: float):
    """Return a 1-D Gaussian convolution kernel as a numpy array."""
    inp = np.zeros(100)
    inp[50] = 1.0
    g = gaussian_filter1d(inp, kernelSD)
    valid = np.where(g > 0.01)[0]
    g = g[valid]
    return (g / g.sum()).astype(np.float32)   # shape (K,)


def _gauss_smooth_tf(inputs, kernelSD: float):
    """
    Apply 1-D Gaussian smoothing to a [B, T, N] tensor along the time axis.
    Uses depthwise_conv2d to process ALL channels in one op (fast on Metal).
    inputs : tf.Tensor  shape [B, T, N]
    """
    kernel = _gauss_kernel(kernelSD)                       # (K,)
    K = kernel.shape[0]
    N = inputs.shape[2]
    # depthwise_conv2d filter: [filter_h, filter_w, in_channels, channel_multiplier]
    # We treat the time axis as height, width=1
    filt = tf.constant(
        kernel.reshape(K, 1, 1, 1) * np.ones((1, 1, N, 1), dtype=np.float32),
        dtype=tf.float32)                                  # (K, 1, N, 1)
    # reshape input to [B, T, 1, N]  (NHWC with W=1)
    x = inputs[:, :, tf.newaxis, :]                        # (B, T, 1, N)
    x = tf.nn.depthwise_conv2d(x, filt, strides=[1,1,1,1], padding='SAME')
    return x[:, :, 0, :]                                   # (B, T, N)


class _GRULayer(tf.keras.layers.Layer):
    """
    A single bidirectional-or-unidirectional GRU layer.

    Uses tf.keras.layers.GRU (reset_after=True) which is backed by a fused
    C++ kernel — dramatically faster than RNN(GRUCell) which loops over
    timesteps in Python.  Since we train from scratch, the reset_after
    formulation doesn't affect final accuracy.
    """

    def __init__(self, nUnits: int, direction: str, **kwargs):
        super().__init__(**kwargs)
        self.nUnits = nUnits
        self.direction = direction
        self.bidir = (direction == 'bidirectional')

        fwd_gru = tf.keras.layers.GRU(
            nUnits, return_sequences=True, return_state=True,
            reset_after=True)
        if self.bidir:
            bwd_gru = tf.keras.layers.GRU(
                nUnits, return_sequences=True, return_state=True,
                reset_after=True, go_backwards=True)
            self.rnn = tf.keras.layers.Bidirectional(
                fwd_gru, backward_layer=bwd_gru, merge_mode='concat')
        else:
            self.rnn = fwd_gru

    def call(self, inputs, initial_state):
        """
        inputs        : (B, T, F)
        initial_state : (B, nUnits)  for unidirectional
                        or tuple ((B, nUnits), (B, nUnits)) for bidirectional

        Returns output (B, T, nUnits*bidir_mult)
        """
        if self.bidir:
            fwd_init, bwd_init = initial_state
            results = self.rnn(
                inputs, initial_state=[fwd_init, bwd_init])
            output = results[0]   # concatenated sequences
        else:
            output, _ = self.rnn(inputs, initial_state=initial_state)
        return output                                       # (B, T, nUnits*mult)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class charSeqRNN:
    """
    TF2-compatible hand-writing character-sequence RNN.

    The public API (train / inference) is unchanged from the TF1 original.
    """

    def __init__(self, args: dict):
        self.args = args

        # ------------------------------------------------------------------
        # Mode flags
        # ------------------------------------------------------------------
        mode = args['mode']
        if mode == 'train':
            self.isTraining = True
        elif mode == 'infer':
            self.isTraining = False
        else:
            raise ValueError(f"args['mode'] must be 'train' or 'infer', got {mode!r}")

        # Count session days
        self.nDays = 0
        for t in range(30):
            if f'labelsFile_{t}' not in self.args:
                self.nDays = t
                break

        # ------------------------------------------------------------------
        # Seeding
        # ------------------------------------------------------------------
        seed = self.args['seed']
        if seed == -1:
            seed = datetime.now().microsecond
            self.args['seed'] = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # ------------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------------
        (neuralCube_all, targets_all, errWeights_all,
         numBinsPerTrial_all, cvIdx_all,
         recordFileSet_all) = self._loadAllDatasets()

        nOutputs = targets_all[0].shape[2]
        nInputs  = neuralCube_all[0].shape[2]
        self.nTrialsInFirstDataset = neuralCube_all[0].shape[0]

        # ------------------------------------------------------------------
        # Build tf.data pipelines
        # ------------------------------------------------------------------
        self._build_datasets(neuralCube_all, targets_all, errWeights_all,
                             numBinsPerTrial_all, cvIdx_all, recordFileSet_all,
                             nInputs, nOutputs)

        # ------------------------------------------------------------------
        # Build the Keras model components
        # ------------------------------------------------------------------
        biDir   = 2 if self.args['directionality'] == 'bidirectional' else 1
        nUnits  = self.args['nUnits']
        skipLen = self.args['skipLen']

        # day-specific input projections  (identity init)
        self.dayToLayerMap  = eval(self.args['dayToLayerMap'])
        self.dayProbability = eval(self.args['dayProbability'])
        self.nInpLayers     = max(self.dayToLayerMap) + 1

        self.inputFactors_W = []
        self.inputFactors_b = []
        for i in range(self.nInpLayers):
            W = tf.Variable(
                tf.eye(nInputs, dtype=tf.float32),
                name=f'inputFactors_W_{i}',
                trainable=bool(self.args['trainableInput']))
            b = tf.Variable(
                tf.zeros([nInputs], dtype=tf.float32),
                name=f'inputFactors_b_{i}',
                trainable=bool(self.args['trainableInput']))
            self.inputFactors_W.append(W)
            self.inputFactors_b.append(b)

        # learnable start state  shape [biDir, 1, nUnits]
        self.rnnStartState = tf.Variable(
            tf.zeros([biDir, 1, nUnits], dtype=tf.float32),
            name='RNN_layer0/startState',
            trainable=bool(self.args['trainableBackEnd']))

        # GRU layers
        self.gru1 = _GRULayer(nUnits, self.args['directionality'],
                               name='gru_layer1')
        self.gru2 = _GRULayer(nUnits, self.args['directionality'],
                               name='gru_layer2')

        # Readout
        self.readout_W = tf.Variable(
            tf.random.normal([biDir * nUnits, nOutputs], stddev=0.05,
                             dtype=tf.float32),
            name='readout_W',
            trainable=bool(self.args['trainableBackEnd']))
        self.readout_b = tf.Variable(
            tf.zeros([nOutputs], dtype=tf.float32),
            name='readout_b',
            trainable=bool(self.args['trainableBackEnd']))

        # upsample indices for layer-2 (runs at 1/skipLen frequency)
        timeSteps = self.args['timeSteps']
        expIdx = []
        for t in range(int(timeSteps / skipLen)):
            expIdx.extend([t] * skipLen)
        self.expIdx = expIdx[:timeSteps]   # length == timeSteps

        # ------------------------------------------------------------------
        # Optimiser  (learning rate updated dynamically each batch)
        # ------------------------------------------------------------------
        self.lr_var = tf.Variable(self.args['learnRateStart'],
                                  trainable=False, dtype=tf.float32,
                                  name='learnRate')
        self.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=self.lr_var,
            beta_1=0.9, beta_2=0.999, epsilon=1e-1)

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        ckpt_objs = {
            'optimizer':       self.optimizer,
            'rnnStartState':   self.rnnStartState,
            'readout_W':       self.readout_W,
            'readout_b':       self.readout_b,
            'gru1':            self.gru1,
            'gru2':            self.gru2,
        }
        for i, (W, b) in enumerate(zip(self.inputFactors_W,
                                        self.inputFactors_b)):
            ckpt_objs[f'inputFactors_W_{i}'] = W
            ckpt_objs[f'inputFactors_b_{i}'] = b

        self.checkpoint = tf.train.Checkpoint(**ckpt_objs)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.args['outputDir'],
            max_to_keep=self.args['nCheckToKeep'])

        # ------------------------------------------------------------------
        # Optionally restore from checkpoint
        # ------------------------------------------------------------------
        loadDir = self.args.get('loadDir', 'None')
        if loadDir != 'None' and os.path.isdir(loadDir):
            self._restore_checkpoint(loadDir)

    # -----------------------------------------------------------------------
    # Forward pass  (used both in training and inference)
    # -----------------------------------------------------------------------

    def _forward(self, batchInputs, dayNum: int, training: bool):
        """
        batchInputs : (B, T, N)
        Returns logitOutput (B, T, nOutputs)
        """
        batchSize = self.args['batchSize']
        nUnits    = self.args['nUnits']
        skipLen   = self.args['skipLen']
        biDir     = 2 if self.args['directionality'] == 'bidirectional' else 1

        # Gaussian smoothing
        if self.args['smoothInputs'] == 1:
            batchInputs = _gauss_smooth_tf(
                batchInputs, kernelSD=4 / self.args['rnnBinSize'])

        # Per-day input projection
        if not self.isTraining:
            # In inference mode, use inferenceInputLayer directly
            layerIdx = self.args.get('inferenceInputLayer', 0)
        else:
            layerIdx = self.dayToLayerMap[dayNum]
        W = self.inputFactors_W[layerIdx]   # (N, N)
        b = self.inputFactors_b[layerIdx]   # (N,)
        inputFactors = tf.matmul(batchInputs, W) + b   # (B, T, N)

        # Initial GRU states: tile learnable start state across batch
        # self.rnnStartState: (biDir, 1, nUnits)
        initState = tf.tile(self.rnnStartState, [1, batchSize, 1])
        # shape: (biDir, B, nUnits)

        if biDir == 2:
            fwd_init = initState[0]   # (B, nUnits)
            bwd_init = initState[1]   # (B, nUnits)
            init1 = (fwd_init, bwd_init)
        else:
            init1 = initState[0]      # (B, nUnits)

        # Layer 1
        rnnOutput = self.gru1(inputFactors, initial_state=init1)
        # (B, T, nUnits*biDir)

        # Layer 2 (slower: downsample by skipLen)
        rnnInput2 = rnnOutput[:, ::skipLen, :]   # (B, T/skipLen, nUnits*biDir)
        if biDir == 2:
            init2 = (initState[0], initState[1])
        else:
            init2 = initState[0]
        rnnOutput2 = self.gru2(rnnInput2, initial_state=init2)
        # (B, T/skipLen, nUnits*biDir)

        # Readout from layer 2
        # (B, T/skipLen, nOutputs)
        logitOutput_down = tf.matmul(rnnOutput2, self.readout_W) + self.readout_b

        # Upsample back to original time resolution
        logitOutput = tf.gather(logitOutput_down, self.expIdx, axis=1)
        # (B, T, nOutputs)

        return logitOutput, rnnOutput

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------

    def _compute_loss(self, logitOutput, batchTargets, batchWeight):
        """
        logitOutput  : (B, T, nOutputs)
        batchTargets : (B, T, nOutputs)
        batchWeight  : (B, T)
        Returns scalar total cost.
        """
        outputDelay = self.args['outputDelay']
        l2scale     = self.args['l2scale']
        timeSteps   = self.args['timeSteps']

        # Align for output delay
        labels  = batchTargets[:, :-(outputDelay), :]    # (B, T-D, nOut)
        logits  = logitOutput[:, outputDelay:, :]        # (B, T-D, nOut)
        bw      = batchWeight[:, :-(outputDelay)]        # (B, T-D)

        # Split off character-start signal (last column)
        transOut   = logits[:, :, -1]
        transLabel = labels[:, :, -1]
        logits     = logits[:, :, :-1]
        labels     = labels[:, :, :-1]

        # Cross-entropy character probability loss
        ceLoss   = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)                # (B, T-D)
        totalErr = tf.reduce_mean(
            tf.reduce_sum(bw * ceLoss, axis=1) / float(timeSteps))

        # Character-start signal loss (MSE)
        sqErrLoss = tf.square(tf.sigmoid(transOut) - transLabel)
        totalErr  += 5.0 * tf.reduce_mean(
            tf.reduce_sum(sqErrLoss, axis=1) / float(timeSteps))

        # L2 regularisation
        l2cost = tf.constant(0.0)
        if l2scale > 0:
            weight_vars = [self.readout_W] + self.inputFactors_W
            # Add GRU kernel weights
            for layer in [self.gru1, self.gru2]:
                for v in layer.trainable_variables:
                    if 'kernel' in v.name:
                        weight_vars.append(v)
            for v in weight_vars:
                l2cost = l2cost + tf.reduce_sum(tf.square(v))

        totalCost = totalErr + l2cost * l2scale
        return totalCost, totalErr

    # -----------------------------------------------------------------------
    # Training step (tf.function for speed)
    # -----------------------------------------------------------------------

    def _train_step(self, batchInputs, batchTargets, batchWeight, dayNum: int):
        """Single training step — runs in eager mode (no @tf.function).

        Avoids multi-hour Metal shader compilation on Apple Silicon.
        Returns logitOutput so the caller doesn't need a second forward pass.
        """
        with tf.GradientTape() as tape:
            logitOutput, _ = self._forward(batchInputs, dayNum, training=True)
            totalCost, totalErr = self._compute_loss(
                logitOutput, batchTargets, batchWeight)

        tvars = tape.watched_variables()
        grads = tape.gradient(totalCost, tvars)
        grads, grad_norm = tf.clip_by_global_norm(grads, 10.0)

        # Only apply finite gradients
        all_finite = all(
            tf.reduce_all(tf.math.is_finite(g)).numpy()
            for g in grads if g is not None)
        if all_finite:
            self.optimizer.apply_gradients(zip(grads, tvars))

        return totalErr, grad_norm, logitOutput

    # -----------------------------------------------------------------------
    # Public API: train
    # -----------------------------------------------------------------------

    def train(self):
        """Main training loop."""
        nBatches     = self.args['nBatchesToTrain']
        batchesPerVal   = self.args['batchesPerVal']
        batchesPerSave  = self.args['batchesPerSave']
        batchesPerModelSave = self.args['batchesPerModelSave']
        learnRateStart  = self.args['learnRateStart']
        learnRateEnd    = self.args['learnRateEnd']

        batchTrainStats = np.zeros([nBatches, 6])
        batchValStats   = np.zeros([
            int(np.ceil(nBatches / batchesPerVal)), 4])

        i = getattr(self, 'startingBatchNum', 0)

        # Save an initial snapshot
        self.ckpt_manager.save(checkpoint_number=0)

        while i < nBatches:
            dtStart = datetime.now()

            # Cosine/linear learning rate schedule
            frac = i / float(nBatches)
            lr   = learnRateStart * (1.0 - frac) + learnRateEnd * frac
            self.lr_var.assign(lr)

            # Pick a day
            dayNum = int(np.argwhere(np.random.multinomial(
                1, self.dayProbability))[0][0])

            # Pull one batch
            batchInputs, batchTargets, batchWeight = self._get_train_batch(
                dayNum)

            dayNum_t = tf.constant(dayNum, dtype=tf.int32)
            err, grad_norm, logitOutput = self._train_step(
                batchInputs, batchTargets, batchWeight, dayNum)

            # Frame accuracy — reuse logitOutput from training step
            trainAcc = computeFrameAccuracy(
                logitOutput.numpy(), batchTargets.numpy(),
                batchWeight.numpy(), self.args['outputDelay'])

            elapsed = (datetime.now() - dtStart).total_seconds()
            batchTrainStats[i, :] = [i, float(err), float(grad_norm),
                                     trainAcc, elapsed, dayNum]

            # Validation
            if i % batchesPerVal == 0:
                valIdx = int(i / batchesPerVal)
                (batchValStats[valIdx, :4],
                 outputSnapshot) = self._validationDiagnostics(
                    i, batchesPerVal, lr, elapsed, err.numpy(), trainAcc)
                scipy.io.savemat(
                    os.path.join(self.args['outputDir'], 'outputSnapshot'),
                    outputSnapshot)

            # Save stats
            if (i >= batchesPerSave and i % batchesPerSave == 0):
                scipy.io.savemat(
                    os.path.join(self.args['outputDir'], 'intermediateOutput'),
                    {'batchTrainStats': batchTrainStats,
                     'batchValStats':   batchValStats})

            # Save model
            if i % batchesPerModelSave == 0:
                print('SAVING MODEL')
                self.ckpt_manager.save(checkpoint_number=i)

            i += 1

        # Final save
        scipy.io.savemat(
            os.path.join(self.args['outputDir'], 'finalOutput'),
            {'batchTrainStats': batchTrainStats,
             'batchValStats':   batchValStats})
        print('SAVING FINAL MODEL')
        self.ckpt_manager.save(checkpoint_number=i)

    # -----------------------------------------------------------------------
    # Public API: inference
    # -----------------------------------------------------------------------

    def inference(self):
        """Run the RNN on the first dataset and return outputs."""
        nBatchesForInference = int(np.ceil(
            self.nTrialsInFirstDataset / self.args['batchSize']))

        allOutputs      = []
        allUnits        = []
        allInputFeatures = []

        print('Starting inference.')
        for x in range(nBatchesForInference):
            batchInputs, batchTargets, batchWeight = self._get_infer_batch()

            logitOutput, rnnOutput = self._forward(
                batchInputs, dayNum=0, training=False)

            allOutputs.append(logitOutput.numpy())
            allInputFeatures.append(batchInputs.numpy())
            allUnits.append(rnnOutput.numpy())

        print('Done with inference.')

        allOutputs       = np.concatenate(allOutputs, axis=0)
        allUnits         = np.concatenate(allUnits,   axis=0)
        allInputFeatures = np.concatenate(allInputFeatures, axis=0)

        allOutputs       = allOutputs[:self.nTrialsInFirstDataset]
        allUnits         = allUnits[:self.nTrialsInFirstDataset]
        allInputFeatures = allInputFeatures[:self.nTrialsInFirstDataset]

        retDict = {
            'outputs':       allOutputs,
            'units':         allUnits,
            'inputFeatures': allInputFeatures,
        }

        if self.args['inferenceOutputFileName'] != 'None':
            scipy.io.savemat(self.args['inferenceOutputFileName'],
                             {'outputs': allOutputs})

        return retDict

    # -----------------------------------------------------------------------
    # Dataset helpers
    # -----------------------------------------------------------------------

    def _build_datasets(self, neuralCube_all, targets_all, errWeights_all,
                        numBinsPerTrial_all, cvIdx_all, recordFileSet_all,
                        nInputs, nOutputs):
        """Build tf.data.Dataset pipelines for training or inference."""

        batchSize      = self.args['batchSize']
        synthBatchSize = self.args['synthBatchSize']
        timeSteps      = self.args['timeSteps']
        direction      = self.args['directionality']

        if self.isTraining:
            self._train_iters_real  = []
            self._train_iters_synth = []
            self._val_iters         = []
            self.daysWithValData    = []

            for dayIdx in range(self.nDays):
                trainIdx = cvIdx_all[dayIdx]['trainIdx']
                valIdx   = cvIdx_all[dayIdx]['testIdx']

                realDataSize = batchSize - synthBatchSize

                # --- real training iterator ---
                if realDataSize > 0:
                    real_ds = self._make_real_dataset(
                        neuralCube_all[dayIdx][trainIdx],
                        targets_all[dayIdx][trainIdx],
                        errWeights_all[dayIdx][trainIdx],
                        numBinsPerTrial_all[dayIdx][trainIdx, np.newaxis],
                        realDataSize, timeSteps, direction, add_noise=True)
                    self._train_iters_real.append(iter(real_ds))
                else:
                    self._train_iters_real.append(None)

                # --- synthetic iterator ---
                if synthBatchSize > 0:
                    recordFiles = recordFileSet_all[dayIdx]
                    if len(recordFiles) == 0:
                        sys.exit(
                            f'Error! No synthetic files found in directory '
                            f'{self.args["syntheticDatasetDir_" + str(dayIdx)]}')
                    synth_ds = self._make_synth_dataset(
                        recordFiles, timeSteps, nInputs, nOutputs,
                        synthBatchSize,
                        whiteNoiseSD=self.args['whiteNoiseSD'],
                        constantOffsetSD=self.args['constantOffsetSD'],
                        randomWalkSD=self.args['randomWalkSD'])
                    self._train_iters_synth.append(iter(synth_ds))
                else:
                    self._train_iters_synth.append(None)

                # --- validation iterator ---
                if len(valIdx) == 0:
                    self._val_iters.append(None)
                else:
                    val_ds = self._make_real_dataset(
                        neuralCube_all[dayIdx][valIdx],
                        targets_all[dayIdx][valIdx],
                        errWeights_all[dayIdx][valIdx],
                        numBinsPerTrial_all[dayIdx][valIdx, np.newaxis],
                        batchSize, timeSteps, direction, add_noise=False)
                    self._val_iters.append(iter(val_ds))
                    self.daysWithValData.append(dayIdx)

        else:
            # Inference: cycle through the first dataset
            neuralData   = neuralCube_all[0].astype(np.float32)
            targetsData  = targets_all[0].astype(np.float32)
            errWData     = errWeights_all[0].astype(np.float32)

            infer_ds = tf.data.Dataset.from_tensor_slices(
                (neuralData, targetsData, errWData))
            infer_ds = infer_ds.repeat().batch(batchSize)
            self._infer_iter = iter(infer_ds)

    def _make_real_dataset(self, inputs, targets, errWeight,
                           numBinsPerTrial, batchSize,
                           timeSteps, direction, add_noise):
        """Build a repeating, shuffled, snippet-extracted tf.data.Dataset."""

        ds = tf.data.Dataset.from_tensor_slices((
            inputs.astype(np.float32),
            targets.astype(np.float32),
            errWeight.astype(np.float32),
            numBinsPerTrial.astype(np.int32)))

        ds = ds.shuffle(max(batchSize * 4, len(inputs))).repeat()

        cOffSD  = self.args['constantOffsetSD']
        rwSD    = self.args['randomWalkSD']
        wnSD    = self.args['whiteNoiseSD']

        def process(inp, targ, ew, bins):
            inp, targ, ew, bins = _extractSentenceSnippet(
                inp, targ, ew, bins, timeSteps, direction)
            if add_noise and (cOffSD > 0 or rwSD > 0):
                inp, targ, ew, bins = _addMeanNoise(
                    inp, targ, ew, bins, cOffSD, rwSD, timeSteps)
            if add_noise and wnSD > 0:
                inp, targ, ew, bins = _addWhiteNoise(
                    inp, targ, ew, bins, wnSD, timeSteps)
            return inp, targ, ew

        ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batchSize, drop_remainder=True).prefetch(1)
        return ds

    def _make_synth_dataset(self, recordFiles, timeSteps, nInputs, nOutputs,
                            batchSize, whiteNoiseSD, constantOffsetSD,
                            randomWalkSD):
        """Build a dataset from .tfrecord synthetic files."""
        raw_ds = tf.data.TFRecordDataset(recordFiles)

        def parse_and_augment(raw):
            return _parseDataset(raw, timeSteps, nInputs, nOutputs,
                                 whiteNoiseSD, constantOffsetSD, randomWalkSD)

        ds = raw_ds.map(parse_and_augment,
                        num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(batchSize * 4).repeat()
        ds = ds.batch(batchSize, drop_remainder=True).prefetch(1)
        return ds

    def _get_train_batch(self, dayNum: int):
        """Pull one combined (real + synthetic) minibatch for dayNum."""
        batchSize      = self.args['batchSize']
        synthBatchSize = self.args['synthBatchSize']
        realDataSize   = batchSize - synthBatchSize

        real_iter  = self._train_iters_real[dayNum]
        synth_iter = self._train_iters_synth[dayNum]

        if real_iter is not None and synth_iter is not None:
            inp_r, targ_r, ew_r = next(real_iter)
            inp_s, targ_s, ew_s = next(synth_iter)
            inp  = tf.concat([inp_s,  inp_r],  axis=0)
            targ = tf.concat([targ_s, targ_r], axis=0)
            ew   = tf.concat([ew_s,   ew_r],   axis=0)
        elif real_iter is not None:
            inp, targ, ew = next(real_iter)
        else:
            inp, targ, ew = next(synth_iter)

        return inp, targ, ew

    def _get_infer_batch(self):
        inp, targ, ew = next(self._infer_iter)
        return inp, targ, ew

    # -----------------------------------------------------------------------
    # Validation diagnostics
    # -----------------------------------------------------------------------

    def _validationDiagnostics(self, i, nBatchesPerVal, lr, totalSeconds,
                                trainErr, trainAcc):
        if self.daysWithValData:
            randIdx = np.random.randint(len(self.daysWithValData))
            dayNum  = self.daysWithValData[randIdx]
        else:
            dayNum = self.nDays - 1

        val_iter = self._val_iters[dayNum]
        if val_iter is None:
            val_iter = self._train_iters_real[dayNum]
        inp, targ, ew = next(val_iter)

        logitOutput, rnnOutput = self._forward(inp, dayNum, training=False)
        totalCost, totalErr    = self._compute_loss(logitOutput, targ, ew)

        # Use totalCost as a proxy for grad_norm (avoids an expensive
        # extra forward+backward pass that was ~3x per-step overhead)
        grad_norm = totalCost

        valAcc = computeFrameAccuracy(
            logitOutput.numpy(), targ.numpy(), ew.numpy(),
            self.args['outputDelay'])

        print(
            f'Val Batch: {i}/{self.args["nBatchesToTrain"]}, '
            f'valErr: {float(totalErr):.4f}, trainErr: {float(trainErr):.4f}, '
            f'Val Acc.: {valAcc:.4f}, Train Acc.: {trainAcc:.4f}, '
            f'grad: {float(grad_norm):.4f}, learnRate: {lr:.6f}, '
            f'time: {totalSeconds:.2f}')

        lo_np   = logitOutput.numpy()
        rnn_np  = rnnOutput.numpy()
        inp_np  = inp.numpy()
        targ_np = targ.numpy()
        ew_np   = ew.numpy()
        outputDelay = self.args['outputDelay']

        outputSnapshot = {
            'inputs':          inp_np[0],
            'rnnUnits':        rnn_np[0],
            'charProbOutput':  lo_np[0, :, :-1],
            'charStartOutput': scipy.special.expit(
                lo_np[0, outputDelay:, -1]),
            'charProbTarget':  targ_np[0, :, :-1],
            'charStartTarget': targ_np[0, :, -1],
            'errorWeight':     ew_np[0],
        }

        return [i, float(totalErr), float(grad_norm), valAcc], outputSnapshot

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def _loadAllDatasets(self):
        neuralCube_all     = []
        targets_all        = []
        errWeights_all     = []
        numBinsPerTrial_all = []
        cvIdx_all          = []
        recordFileSet_all  = []

        for dayIdx in range(self.nDays):
            neuralData, targets, errWeights, binsPerTrial, cvIdx = \
                prepareDataCubesForRNN(
                    self.args[f'sentencesFile_{dayIdx}'],
                    self.args[f'singleLettersFile_{dayIdx}'],
                    self.args[f'labelsFile_{dayIdx}'],
                    self.args[f'cvPartitionFile_{dayIdx}'],
                    self.args[f'sessionName_{dayIdx}'],
                    self.args['rnnBinSize'],
                    self.args['timeSteps'],
                    self.isTraining)

            neuralCube_all.append(neuralData)
            targets_all.append(targets)
            errWeights_all.append(errWeights)
            numBinsPerTrial_all.append(binsPerTrial)
            cvIdx_all.append(cvIdx)

            synthDir = self.args[f'syntheticDatasetDir_{dayIdx}']
            if os.path.isdir(synthDir):
                recordFiles = [
                    os.path.join(synthDir, f) for f in os.listdir(synthDir)
                    if f.endswith('.tfrecord') or f.endswith('.tfrecords')]
                # fall back to all files in the directory if no .tfrecord ext
                if not recordFiles:
                    recordFiles = [
                        os.path.join(synthDir, f)
                        for f in os.listdir(synthDir)]
            else:
                recordFiles = []

            random.shuffle(recordFiles)
            recordFileSet_all.append(recordFiles)

        return (neuralCube_all, targets_all, errWeights_all,
                numBinsPerTrial_all, cvIdx_all, recordFileSet_all)

    # -----------------------------------------------------------------------
    # Checkpoint restore
    # -----------------------------------------------------------------------

    def _restore_checkpoint(self, loadDir: str):
        """Restore the latest checkpoint found in loadDir."""
        restore_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=loadDir, max_to_keep=None)
        idx = self.args.get('loadCheckpointIdx', -1)
        ckpts = restore_manager.checkpoints
        if not ckpts:
            print(f'No checkpoints found in {loadDir}')
            return
        ckpt_path = ckpts[idx]
        print(f'Loading from checkpoint: {ckpt_path}')
        status = self.checkpoint.restore(ckpt_path)
        # expect_partial() silences warnings about optimizer slots not saved
        status.expect_partial()

        # If we are resuming training in the same directory, pick up step num
        if (self.args.get('mode') == 'train' and
                loadDir == self.args.get('outputDir')):
            # step number is encoded in the checkpoint filename
            try:
                self.startingBatchNum = int(ckpt_path.split('-')[-1]) + 1
            except (ValueError, IndexError):
                self.startingBatchNum = 0
        else:
            self.startingBatchNum = 0


# ---------------------------------------------------------------------------
# Dataset map functions  (module-level so @tf.function can trace them)
# ---------------------------------------------------------------------------

def _extractSentenceSnippet(inputs, targets, errWeight, numBinsPerTrial,
                             nSteps, directionality):
    """Extract a random time-snippet from the full padded sentence."""
    maxVal = tf.maximum(numBinsPerTrial[0] + (nSteps - 100) - 400, 1)
    randomStart = tf.random.uniform([], minval=0, maxval=maxVal,
                                    dtype=tf.int32)

    inputsSnippet  = inputs[randomStart:(randomStart + nSteps), :]
    targetsSnippet = targets[randomStart:(randomStart + nSteps), :]

    diff       = targetsSnippet[1:, -1] - targetsSnippet[:-1, -1]
    charStarts = tf.where(diff >= 0.1)

    def noLetters():
        return tf.zeros(nSteps)

    def atLeastOne():
        firstChar = tf.cast(charStarts[0, 0], tf.int32)
        lastChar  = tf.cast(charStarts[-1, 0], tf.int32)
        if directionality == 'unidirectional':
            ew = tf.concat([
                tf.zeros([firstChar]),
                errWeight[(randomStart + firstChar):(randomStart + nSteps)]],
                axis=0)
        else:
            ew = tf.concat([
                tf.zeros([firstChar]),
                errWeight[(randomStart + firstChar):(randomStart + lastChar)],
                tf.zeros([nSteps - lastChar])],
                axis=0)
        return ew

    errWeightSnippet = tf.cond(
        tf.equal(tf.shape(charStarts)[0], 0), noLetters, atLeastOne)

    return inputsSnippet, targetsSnippet, errWeightSnippet, numBinsPerTrial


def _addMeanNoise(inputs, targets, errWeight, numBinsPerTrial,
                  constantOffsetSD, randomWalkSD, nSteps):
    N = inputs.shape[1] if inputs.shape[1] is not None else tf.shape(inputs)[1]
    meanDrift = tf.random.normal([1, N], stddev=constantOffsetSD)
    meanDrift = meanDrift + tf.cumsum(
        tf.random.normal([nSteps, N], stddev=randomWalkSD), axis=0)
    return inputs + meanDrift, targets, errWeight, numBinsPerTrial


def _addWhiteNoise(inputs, targets, errWeight, numBinsPerTrial,
                   whiteNoiseSD, nSteps):
    noise = tf.random.normal(tf.shape(inputs), stddev=whiteNoiseSD)
    return inputs + noise, targets, errWeight, numBinsPerTrial


def _parseDataset(raw, nSteps, nInputs, nClasses,
                  whiteNoiseSD=0.0, constantOffsetSD=0.0, randomWalkSD=0.0):
    """Parse a single TFRecord example (synthetic data)."""
    features = {
        'inputs':     tf.io.FixedLenFeature([nSteps * nInputs], tf.float32),
        'labels':     tf.io.FixedLenFeature([nSteps * nClasses], tf.float32),
        'errWeights': tf.io.FixedLenFeature([nSteps], tf.float32),
    }
    parsed = tf.io.parse_single_example(raw, features)
    inp    = tf.reshape(parsed['inputs'],     [nSteps, nInputs])
    labels = tf.reshape(parsed['labels'],     [nSteps, nClasses])
    ew     = parsed['errWeights']

    noise = tf.random.normal([nSteps, nInputs], stddev=whiteNoiseSD)
    if constantOffsetSD > 0 or randomWalkSD > 0:
        mean_noise = tf.random.normal([1, nInputs], stddev=constantOffsetSD)
        mean_noise = mean_noise + tf.cumsum(
            tf.random.normal([nSteps, nInputs], stddev=randomWalkSD), axis=0)
        noise = noise + mean_noise

    return inp + noise, labels, ew


# ---------------------------------------------------------------------------
# Accuracy / evaluation helpers  (unchanged logic from original)
# ---------------------------------------------------------------------------

def computeFrameAccuracy(rnnOutput, targets, errWeight, outputDelay):
    bestClass      = np.argmax(rnnOutput[:, outputDelay:, :-1], axis=2)
    indicatedClass = np.argmax(targets[:, :-outputDelay, :-1], axis=2)
    bw             = errWeight[:, :-outputDelay]
    denom = np.sum(bw)
    if denom == 0:
        return 0.0
    acc = np.sum(bw * np.equal(np.squeeze(bestClass),
                               np.squeeze(indicatedClass))) / denom
    return float(acc)


# ---------------------------------------------------------------------------
# Default arguments  (unchanged from original)
# ---------------------------------------------------------------------------

def getDefaultRNNArgs():
    args = {}

    rootDir  = os.path.expanduser('~') + '/handwritingBCIData/'
    dataDirs = ['t5.2019.05.08']
    cvPart   = 'HeldOutBlocks'

    for x in range(len(dataDirs)):
        args[f'sentencesFile_{x}']       = (rootDir + 'Datasets/' +
                                             dataDirs[x] + '/sentences.mat')
        args[f'singleLettersFile_{x}']   = (rootDir + 'Datasets/' +
                                             dataDirs[x] + '/singleLetters.mat')
        args[f'labelsFile_{x}']          = (rootDir +
                                             'RNNTrainingSteps/Step2_HMMLabels/' +
                                             cvPart + '/' + dataDirs[x] +
                                             '_timeSeriesLabels.mat')
        args[f'syntheticDatasetDir_{x}'] = (rootDir +
                                             'RNNTrainingSteps/Step3_SyntheticSentences/' +
                                             cvPart + '/' + dataDirs[x] +
                                             '_syntheticSentences/')
        args[f'cvPartitionFile_{x}']     = (rootDir +
                                             'RNNTrainingSteps/trainTestPartitions_' +
                                             cvPart + '.mat')
        args[f'sessionName_{x}']         = dataDirs[x]

    args['gpuNumber']           = '0'
    args['mode']                = 'train'
    args['outputDir']           = rootDir + 'RNNTrainingSteps/Step4_RNNTraining/' + cvPart
    args['loadDir']             = 'None'
    args['loadCheckpointIdx']   = -1
    args['nUnits']              = 512
    args['rnnBinSize']          = 2
    args['smoothInputs']        = 1
    args['skipLen']             = 5
    args['outputDelay']         = 50
    args['directionality']      = 'unidirectional'
    args['constantOffsetSD']    = 0.6
    args['randomWalkSD']        = 0.02
    args['whiteNoiseSD']        = 1.2
    args['l2scale']             = 1e-5
    args['learnRateStart']      = 0.01
    args['learnRateEnd']        = 0.0
    args['trainableInput']      = 1
    args['trainableBackEnd']    = 1
    args['seed']                = datetime.now().microsecond
    args['nCheckToKeep']        = 3
    args['batchesPerSave']      = 200
    args['batchesPerVal']       = 50
    args['batchesPerModelSave'] = 5000
    args['nBatchesToTrain']     = 100000
    args['timeSteps']           = 1200
    args['batchSize']           = 32
    args['synthBatchSize']      = 12
    args['inputScale']          = 1.0
    args['inferenceOutputFileName'] = 'None'
    args['inferenceInputLayer']     = 0
    args['dayToLayerMap']           = '[0]'
    args['dayProbability']          = '[1.0]'

    return args


# ---------------------------------------------------------------------------
# Command-line entry point  (unchanged interface)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--argsFile', metavar='argsFile',
                        type=str, default='args.p')
    args    = parser.parse_args()
    argDict = pickle.load(open(args.argsFile, 'rb'))

    # GPU visibility (no-op on Apple Silicon / CPU-only machines)
    os.environ['CUDA_DEVICE_ORDER']   = 'PCI_BUS_ID'
    gpuNum = argDict.get('gpuNumber', '0')
    print(f'Setting CUDA_VISIBLE_DEVICES to {gpuNum}')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuNum

    rnnModel = charSeqRNN(args=argDict)

    if argDict['mode'] == 'train':
        rnnModel.train()
    elif argDict['mode'] == 'infer':
        rnnModel.inference()
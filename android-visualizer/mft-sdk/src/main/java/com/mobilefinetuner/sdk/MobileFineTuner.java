package com.mobilefinetuner.sdk;

import java.io.File;
import java.util.Arrays;

/**
 * Java-facing Android SDK for MobileFineTuner.
 *
 * <p>The SDK owns one native MF session at a time: model, optional LoRA
 * adapters, and an optimizer-backed trainer. Model weights are supplied by the
 * application as files on device storage; they are not bundled in the AAR.</p>
 */
public final class MobileFineTuner implements AutoCloseable {
    static {
        System.loadLibrary("mobilefinetuner_jni");
    }

    private long nativeHandle;

    private MobileFineTuner(long nativeHandle) {
        if (nativeHandle == 0L) {
            throw new IllegalStateException("Native MobileFineTuner session was not created");
        }
        this.nativeHandle = nativeHandle;
    }

    public static String buildInfo() {
        return nativeBuildInfo();
    }

    public static SelfTestResult selfTest(File workingDir) {
        if (workingDir == null) {
            throw new IllegalArgumentException("workingDir must not be null");
        }
        double[] result = nativeSelfTest(workingDir.getAbsolutePath());
        return new SelfTestResult((float) result[0], (int) result[1], result[2]);
    }

    public static MobileFineTuner open(String modelDir) {
        return open(modelDir, true);
    }

    public static MobileFineTuner open(String modelDir, boolean loadWeights) {
        if (modelDir == null || modelDir.isEmpty()) {
            throw new IllegalArgumentException("modelDir must not be empty");
        }
        return new MobileFineTuner(nativeCreate(modelDir, loadWeights));
    }

    public void initLora(LoraConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        nativeInitLora(
                requireOpen(),
                config.rank,
                config.alpha,
                config.dropout,
                config.seed,
                config.targetModules
        );
    }

    public void createTrainer(TrainerConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        nativeCreateTrainer(
                requireOpen(),
                config.learningRate,
                config.weightDecay,
                config.maxGradNorm,
                config.ignoreIndex,
                config.useStreamingLmLoss,
                config.gradientAccumulationSteps
        );
    }

    public void createDpoTrainer(DpoTrainerConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        nativeCreateDpoTrainer(
                requireOpen(),
                config.learningRate,
                config.weightDecay,
                config.maxGradNorm,
                config.beta,
                config.useStreamingDpoLoss,
                config.gradientAccumulationSteps
        );
    }

    public void createKtoTrainer(KtoTrainerConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        nativeCreateKtoTrainer(
                requireOpen(),
                config.learningRate,
                config.weightDecay,
                config.maxGradNorm,
                config.beta,
                config.desirableWeight,
                config.undesirableWeight,
                config.useStreamingKtoLoss,
                config.gradientAccumulationSteps
        );
    }

    public TrainStepResult trainStep(
            int[] inputIds,
            float[] attentionMask,
            int[] labels,
            int batchSize,
            int sequenceLength
    ) {
        if (batchSize <= 0 || sequenceLength <= 1) {
            throw new IllegalArgumentException("batchSize must be positive and sequenceLength must be > 1");
        }
        int expected = batchSize * sequenceLength;
        requireLength(inputIds, expected, "inputIds");
        requireLength(attentionMask, expected, "attentionMask");
        requireLength(labels, expected, "labels");

        double[] result = nativeTrainStep(
                requireOpen(),
                inputIds,
                attentionMask,
                labels,
                batchSize,
                sequenceLength
        );
        return new TrainStepResult(
                (float) result[0],
                (int) result[1],
                (int) result[2],
                (float) result[3],
                (int) result[4],
                (int) result[5],
                result[6] != 0.0
        );
    }

    public TrainStepResult trainTextBatch(String[] texts, int sequenceLength) {
        return trainTextBatch(texts, sequenceLength, false);
    }

    public TrainStepResult trainTextBatch(String[] texts, int sequenceLength, boolean appendEos) {
        if (texts == null || texts.length == 0) {
            throw new IllegalArgumentException("texts must contain at least one item");
        }
        if (sequenceLength <= 1) {
            throw new IllegalArgumentException("sequenceLength must be > 1");
        }
        String[] copied = Arrays.copyOf(texts, texts.length);
        for (String text : copied) {
            if (text == null) {
                throw new IllegalArgumentException("texts must not contain null entries");
            }
        }
        double[] result = nativeTrainTextBatch(requireOpen(), copied, sequenceLength, appendEos);
        return new TrainStepResult(
                (float) result[0],
                (int) result[1],
                (int) result[2],
                (float) result[3],
                (int) result[4],
                (int) result[5],
                result[6] != 0.0
        );
    }

    public PreferenceStepResult trainPreferenceBatch(
            String[] prompts,
            String[] chosen,
            String[] rejected,
            float[] refChosenLogps,
            float[] refRejectedLogps,
            int sequenceLength
    ) {
        return trainPreferenceBatch(
                prompts,
                chosen,
                rejected,
                refChosenLogps,
                refRejectedLogps,
                sequenceLength,
                false
        );
    }

    /**
     * Trains one canonical KTO batch of independently labelled completions.
     *
     * <p>The batch must contain at least two samples. The native batch builder
     * estimates KL with a second branch that combines each prompt with the
     * previous sample's completion. Consequently {@code refLogps} and
     * {@code refKlLogps} must have been produced for this exact batch ordering
     * and tokenization configuration, for example by {@link #scoreKtoBatch}.</p>
     *
     * @param desirable {@code true} for desirable feedback and {@code false}
     *                   for undesirable feedback
     */
    public KtoStepResult trainKtoBatch(
            String[] prompts,
            String[] completions,
            boolean[] desirable,
            float[] refLogps,
            float[] refKlLogps,
            int sequenceLength
    ) {
        return trainKtoBatch(
                prompts,
                completions,
                desirable,
                refLogps,
                refKlLogps,
                sequenceLength,
                false
        );
    }

    /**
     * Trains one canonical KTO batch of independently labelled completions.
     * See {@link #trainKtoBatch(String[], String[], boolean[], float[], float[], int)}.
     */
    public KtoStepResult trainKtoBatch(
            String[] prompts,
            String[] completions,
            boolean[] desirable,
            float[] refLogps,
            float[] refKlLogps,
            int sequenceLength,
            boolean appendEosToCompletion
    ) {
        int batchSize = requireKtoBatch(prompts, completions, desirable);
        requireLength(refLogps, batchSize, "refLogps");
        requireLength(refKlLogps, batchSize, "refKlLogps");
        if (sequenceLength <= 1) {
            throw new IllegalArgumentException("sequenceLength must be > 1");
        }

        double[] result = nativeTrainKtoBatch(
                requireOpen(),
                Arrays.copyOf(prompts, prompts.length),
                Arrays.copyOf(completions, completions.length),
                Arrays.copyOf(desirable, desirable.length),
                Arrays.copyOf(refLogps, refLogps.length),
                Arrays.copyOf(refKlLogps, refKlLogps.length),
                sequenceLength,
                appendEosToCompletion
        );
        return parseKtoStepResult(result);
    }

    /**
     * Legacy paired KTO approximation retained for source compatibility.
     *
     * @deprecated Use {@link #trainKtoBatch} with independent binary feedback.
     */
    @Deprecated
    public KtoStepResult trainKtoPreferenceBatch(
            String[] prompts,
            String[] chosen,
            String[] rejected,
            float[] refChosenLogps,
            float[] refRejectedLogps,
            int sequenceLength
    ) {
        return trainKtoPreferenceBatch(
                prompts,
                chosen,
                rejected,
                refChosenLogps,
                refRejectedLogps,
                sequenceLength,
                false
        );
    }

    /**
     * Legacy paired KTO approximation retained for source compatibility.
     *
     * @deprecated Use {@link #trainKtoBatch} with independent binary feedback.
     */
    @Deprecated
    public KtoStepResult trainKtoPreferenceBatch(
            String[] prompts,
            String[] chosen,
            String[] rejected,
            float[] refChosenLogps,
            float[] refRejectedLogps,
            int sequenceLength,
            boolean appendEosToResponse
    ) {
        int batchSize = requireSameStringBatch(prompts, chosen, rejected);
        requireLength(refChosenLogps, batchSize, "refChosenLogps");
        requireLength(refRejectedLogps, batchSize, "refRejectedLogps");
        if (sequenceLength <= 1) {
            throw new IllegalArgumentException("sequenceLength must be > 1");
        }

        double[] result = nativeTrainKtoPreferenceBatch(
                requireOpen(),
                Arrays.copyOf(prompts, prompts.length),
                Arrays.copyOf(chosen, chosen.length),
                Arrays.copyOf(rejected, rejected.length),
                Arrays.copyOf(refChosenLogps, refChosenLogps.length),
                Arrays.copyOf(refRejectedLogps, refRejectedLogps.length),
                sequenceLength,
                appendEosToResponse
        );
        return parseKtoStepResult(result);
    }

    public PreferenceStepResult trainPreferenceBatch(
            String[] prompts,
            String[] chosen,
            String[] rejected,
            float[] refChosenLogps,
            float[] refRejectedLogps,
            int sequenceLength,
            boolean appendEosToResponse
    ) {
        int batchSize = requireSameStringBatch(prompts, chosen, rejected);
        requireLength(refChosenLogps, batchSize, "refChosenLogps");
        requireLength(refRejectedLogps, batchSize, "refRejectedLogps");
        if (sequenceLength <= 1) {
            throw new IllegalArgumentException("sequenceLength must be > 1");
        }

        double[] result = nativeTrainPreferenceBatch(
                requireOpen(),
                Arrays.copyOf(prompts, prompts.length),
                Arrays.copyOf(chosen, chosen.length),
                Arrays.copyOf(rejected, rejected.length),
                Arrays.copyOf(refChosenLogps, refChosenLogps.length),
                Arrays.copyOf(refRejectedLogps, refRejectedLogps.length),
                sequenceLength,
                appendEosToResponse
        );
        return new PreferenceStepResult(
                (float) result[0],
                (int) result[1],
                (int) result[2],
                (int) result[3],
                (float) result[4],
                (float) result[5],
                (float) result[6],
                (float) result[7],
                (float) result[8],
                (int) result[9],
                (int) result[10],
                result[11] != 0.0
        );
    }

    public int trainableTensorCount() {
        return nativeTrainableTensorCount(requireOpen());
    }

    public PreferenceScoreResult scorePreferenceBatch(
            String[] prompts,
            String[] chosen,
            String[] rejected,
            int sequenceLength,
            boolean appendEosToResponse
    ) {
        int batchSize = requireSameStringBatch(prompts, chosen, rejected);
        if (sequenceLength <= 1) {
            throw new IllegalArgumentException("sequenceLength must be > 1");
        }
        double[] result = nativeScorePreferenceBatch(
                requireOpen(),
                Arrays.copyOf(prompts, prompts.length),
                Arrays.copyOf(chosen, chosen.length),
                Arrays.copyOf(rejected, rejected.length),
                sequenceLength,
                appendEosToResponse
        );
        if (result.length != 1 + batchSize * 2) {
            throw new IllegalStateException("Native score result has unexpected length");
        }
        float[] chosenLogps = new float[batchSize];
        float[] rejectedLogps = new float[batchSize];
        for (int i = 0; i < batchSize; ++i) {
            chosenLogps[i] = (float) result[1 + i];
            rejectedLogps[i] = (float) result[1 + batchSize + i];
        }
        return new PreferenceScoreResult((int) result[0], chosenLogps, rejectedLogps);
    }

    public KtoScoreResult scoreKtoBatch(
            String[] prompts,
            String[] completions,
            boolean[] desirable,
            int sequenceLength
    ) {
        return scoreKtoBatch(prompts, completions, desirable, sequenceLength, false);
    }

    /**
     * Scores both branches needed for cached-reference canonical KTO training.
     *
     * <p>{@link KtoScoreResult#responseLogps} contains scores for matched
     * prompt/completion rows. {@link KtoScoreResult#klResponseLogps} contains
     * scores for the same prompts paired with the previous completion in this
     * batch. Keep the same row order when passing both arrays to
     * {@link #trainKtoBatch}.</p>
     */
    public KtoScoreResult scoreKtoBatch(
            String[] prompts,
            String[] completions,
            boolean[] desirable,
            int sequenceLength,
            boolean appendEosToCompletion
    ) {
        int batchSize = requireKtoBatch(prompts, completions, desirable);
        if (sequenceLength <= 1) {
            throw new IllegalArgumentException("sequenceLength must be > 1");
        }
        double[] result = nativeScoreKtoBatch(
                requireOpen(),
                Arrays.copyOf(prompts, prompts.length),
                Arrays.copyOf(completions, completions.length),
                Arrays.copyOf(desirable, desirable.length),
                sequenceLength,
                appendEosToCompletion
        );
        if (result.length != 2 + batchSize * 2) {
            throw new IllegalStateException("Native KTO score result has unexpected length");
        }
        float[] responseLogps = new float[batchSize];
        float[] klResponseLogps = new float[batchSize];
        for (int i = 0; i < batchSize; ++i) {
            responseLogps[i] = (float) result[2 + i];
            klResponseLogps[i] = (float) result[2 + batchSize + i];
        }
        return new KtoScoreResult(
                (int) result[0],
                (int) result[1],
                responseLogps,
                klResponseLogps
        );
    }

    public void saveLoraAdapter(File path) {
        if (path == null) {
            throw new IllegalArgumentException("path must not be null");
        }
        nativeSaveLoraAdapter(requireOpen(), path.getAbsolutePath());
    }

    public void loadLoraAdapter(File path) {
        if (path == null) {
            throw new IllegalArgumentException("path must not be null");
        }
        nativeLoadLoraAdapter(requireOpen(), path.getAbsolutePath());
    }

    public String generate(String prompt, int maxNewTokens) {
        return generate(prompt, maxNewTokens, 0.0f, 1.0f, 42L);
    }

    public String generate(String prompt, int maxNewTokens, float temperature, float topP, long seed) {
        if (prompt == null || prompt.isEmpty()) {
            throw new IllegalArgumentException("prompt must not be empty");
        }
        if (maxNewTokens <= 0) {
            throw new IllegalArgumentException("maxNewTokens must be positive");
        }
        if (temperature < 0.0f) {
            throw new IllegalArgumentException("temperature must be non-negative");
        }
        if (topP <= 0.0f || topP > 1.0f) {
            throw new IllegalArgumentException("topP must be in (0, 1]");
        }
        return nativeGenerate(requireOpen(), prompt, maxNewTokens, temperature, topP, seed);
    }

    @Override
    public void close() {
        long handle = nativeHandle;
        nativeHandle = 0L;
        if (handle != 0L) {
            nativeClose(handle);
        }
    }

    private long requireOpen() {
        if (nativeHandle == 0L) {
            throw new IllegalStateException("MobileFineTuner session is closed");
        }
        return nativeHandle;
    }

    private static void requireLength(Object array, int expected, String name) {
        if (array == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
        int length;
        if (array instanceof int[]) {
            length = ((int[]) array).length;
        } else if (array instanceof float[]) {
            length = ((float[]) array).length;
        } else {
            throw new IllegalArgumentException(name + " has unsupported array type");
        }
        if (length != expected) {
            throw new IllegalArgumentException(
                    name + " length must be " + expected + ", got " + length
            );
        }
    }

    private static int requireSameStringBatch(String[] prompts, String[] chosen, String[] rejected) {
        if (prompts == null || chosen == null || rejected == null) {
            throw new IllegalArgumentException("preference text arrays must not be null");
        }
        if (prompts.length == 0) {
            throw new IllegalArgumentException("preference batch must contain at least one pair");
        }
        if (chosen.length != prompts.length || rejected.length != prompts.length) {
            throw new IllegalArgumentException("preference text arrays must have the same length");
        }
        for (int i = 0; i < prompts.length; ++i) {
            if (prompts[i] == null || chosen[i] == null || rejected[i] == null) {
                throw new IllegalArgumentException("preference text arrays must not contain null entries");
            }
        }
        return prompts.length;
    }

    private static int requireKtoBatch(
            String[] prompts,
            String[] completions,
            boolean[] desirable
    ) {
        if (prompts == null || completions == null || desirable == null) {
            throw new IllegalArgumentException("KTO arrays must not be null");
        }
        if (prompts.length < 2) {
            throw new IllegalArgumentException("KTO batch must contain at least two samples");
        }
        if (completions.length != prompts.length || desirable.length != prompts.length) {
            throw new IllegalArgumentException("KTO arrays must have the same length");
        }
        for (int i = 0; i < prompts.length; ++i) {
            if (prompts[i] == null || completions[i] == null) {
                throw new IllegalArgumentException("KTO text arrays must not contain null entries");
            }
        }
        return prompts.length;
    }

    private static KtoStepResult parseKtoStepResult(double[] result) {
        if (result == null || (result.length != 13 && result.length != 17)) {
            throw new IllegalStateException("Native KTO train result has unexpected length");
        }
        int pairCount = (int) result[2];
        int sampleCount = (int) result[3];
        return new KtoStepResult(
                (float) result[0],
                (int) result[1],
                pairCount,
                sampleCount,
                (int) result[4],
                (float) result[5],
                (float) result[6],
                (float) result[7],
                (float) result[8],
                (float) result[9],
                (int) result[10],
                (int) result[11],
                result[12] != 0.0,
                result.length == 17 ? (float) result[13] : 0.0f,
                result.length == 17 ? (int) result[14] : pairCount,
                result.length == 17 ? (int) result[15] : pairCount,
                result.length == 17 ? (int) result[16] : 0
        );
    }

    private static native String nativeBuildInfo();
    private static native double[] nativeSelfTest(String workingDir);
    private static native long nativeCreate(String modelDir, boolean loadWeights);
    private static native void nativeInitLora(
            long handle,
            int rank,
            float alpha,
            float dropout,
            long seed,
            String[] targetModules
    );
    private static native void nativeCreateTrainer(
            long handle,
            float learningRate,
            float weightDecay,
            float maxGradNorm,
            int ignoreIndex,
            boolean useStreamingLmLoss,
            int gradientAccumulationSteps
    );
    private static native void nativeCreateDpoTrainer(
            long handle,
            float learningRate,
            float weightDecay,
            float maxGradNorm,
            float beta,
            boolean useStreamingDpoLoss,
            int gradientAccumulationSteps
    );
    private static native void nativeCreateKtoTrainer(
            long handle,
            float learningRate,
            float weightDecay,
            float maxGradNorm,
            float beta,
            float desirableWeight,
            float undesirableWeight,
            boolean useStreamingKtoLoss,
            int gradientAccumulationSteps
    );
    private static native double[] nativeTrainStep(
            long handle,
            int[] inputIds,
            float[] attentionMask,
            int[] labels,
            int batchSize,
            int sequenceLength
    );
    private static native double[] nativeTrainTextBatch(
            long handle,
            String[] texts,
            int sequenceLength,
            boolean appendEos
    );
    private static native double[] nativeTrainPreferenceBatch(
            long handle,
            String[] prompts,
            String[] chosen,
            String[] rejected,
            float[] refChosenLogps,
            float[] refRejectedLogps,
            int sequenceLength,
            boolean appendEosToResponse
    );
    private static native double[] nativeTrainKtoPreferenceBatch(
            long handle,
            String[] prompts,
            String[] chosen,
            String[] rejected,
            float[] refChosenLogps,
            float[] refRejectedLogps,
            int sequenceLength,
            boolean appendEosToResponse
    );
    private static native double[] nativeTrainKtoBatch(
            long handle,
            String[] prompts,
            String[] completions,
            boolean[] desirable,
            float[] refLogps,
            float[] refKlLogps,
            int sequenceLength,
            boolean appendEosToCompletion
    );
    private static native double[] nativeScorePreferenceBatch(
            long handle,
            String[] prompts,
            String[] chosen,
            String[] rejected,
            int sequenceLength,
            boolean appendEosToResponse
    );
    private static native double[] nativeScoreKtoBatch(
            long handle,
            String[] prompts,
            String[] completions,
            boolean[] desirable,
            int sequenceLength,
            boolean appendEosToCompletion
    );
    private static native void nativeSaveLoraAdapter(long handle, String path);
    private static native void nativeLoadLoraAdapter(long handle, String path);
    private static native int nativeTrainableTensorCount(long handle);
    private static native String nativeGenerate(
            long handle,
            String prompt,
            int maxNewTokens,
            float temperature,
            float topP,
            long seed
    );
    private static native void nativeClose(long handle);

    public static final class LoraConfig {
        public final int rank;
        public final float alpha;
        public final float dropout;
        public final long seed;
        public final String[] targetModules;

        public LoraConfig(int rank, float alpha, float dropout, long seed, String[] targetModules) {
            if (rank <= 0) {
                throw new IllegalArgumentException("rank must be positive");
            }
            if (alpha <= 0.0f) {
                throw new IllegalArgumentException("alpha must be positive");
            }
            if (dropout < 0.0f || dropout >= 1.0f) {
                throw new IllegalArgumentException("dropout must be in [0, 1)");
            }
            this.rank = rank;
            this.alpha = alpha;
            this.dropout = dropout;
            this.seed = seed;
            this.targetModules = targetModules == null
                    ? new String[0]
                    : Arrays.copyOf(targetModules, targetModules.length);
        }

        public static LoraConfig attentionQkvo() {
            return new LoraConfig(
                    8,
                    16.0f,
                    0.05f,
                    42L,
                    new String[]{"q_proj", "k_proj", "v_proj", "o_proj"}
            );
        }
    }

    public static final class DpoTrainerConfig {
        public final float learningRate;
        public final float weightDecay;
        public final float maxGradNorm;
        public final float beta;
        public final boolean useStreamingDpoLoss;
        public final int gradientAccumulationSteps;

        public DpoTrainerConfig(float learningRate, float weightDecay, float maxGradNorm, float beta) {
            this(learningRate, weightDecay, maxGradNorm, beta, true, 1);
        }

        public DpoTrainerConfig(
                float learningRate,
                float weightDecay,
                float maxGradNorm,
                float beta,
                boolean useStreamingDpoLoss,
                int gradientAccumulationSteps
        ) {
            if (learningRate <= 0.0f) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            if (weightDecay < 0.0f) {
                throw new IllegalArgumentException("weightDecay must be non-negative");
            }
            if (maxGradNorm <= 0.0f) {
                throw new IllegalArgumentException("maxGradNorm must be positive");
            }
            if (beta <= 0.0f) {
                throw new IllegalArgumentException("beta must be positive");
            }
            if (gradientAccumulationSteps <= 0) {
                throw new IllegalArgumentException("gradientAccumulationSteps must be positive");
            }
            this.learningRate = learningRate;
            this.weightDecay = weightDecay;
            this.maxGradNorm = maxGradNorm;
            this.beta = beta;
            this.useStreamingDpoLoss = useStreamingDpoLoss;
            this.gradientAccumulationSteps = gradientAccumulationSteps;
        }

        public static DpoTrainerConfig defaults() {
            return new DpoTrainerConfig(2e-4f, 0.0f, 1.0f, 0.1f);
        }
    }

    public static final class KtoTrainerConfig {
        public final float learningRate;
        public final float weightDecay;
        public final float maxGradNorm;
        public final float beta;
        public final float desirableWeight;
        public final float undesirableWeight;
        public final boolean useStreamingKtoLoss;
        public final int gradientAccumulationSteps;

        public KtoTrainerConfig(float learningRate, float weightDecay, float maxGradNorm, float beta) {
            this(learningRate, weightDecay, maxGradNorm, beta, 1.0f, 1.0f, true, 1);
        }

        public KtoTrainerConfig(
                float learningRate,
                float weightDecay,
                float maxGradNorm,
                float beta,
                float desirableWeight,
                float undesirableWeight,
                boolean useStreamingKtoLoss,
                int gradientAccumulationSteps
        ) {
            if (learningRate <= 0.0f) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            if (weightDecay < 0.0f) {
                throw new IllegalArgumentException("weightDecay must be non-negative");
            }
            if (maxGradNorm <= 0.0f) {
                throw new IllegalArgumentException("maxGradNorm must be positive");
            }
            if (beta <= 0.0f) {
                throw new IllegalArgumentException("beta must be positive");
            }
            if (desirableWeight <= 0.0f || undesirableWeight <= 0.0f) {
                throw new IllegalArgumentException("KTO weights must be positive");
            }
            if (gradientAccumulationSteps <= 0) {
                throw new IllegalArgumentException("gradientAccumulationSteps must be positive");
            }
            this.learningRate = learningRate;
            this.weightDecay = weightDecay;
            this.maxGradNorm = maxGradNorm;
            this.beta = beta;
            this.desirableWeight = desirableWeight;
            this.undesirableWeight = undesirableWeight;
            this.useStreamingKtoLoss = useStreamingKtoLoss;
            this.gradientAccumulationSteps = gradientAccumulationSteps;
        }

        public static KtoTrainerConfig defaults() {
            return new KtoTrainerConfig(2e-4f, 0.0f, 1.0f, 0.1f);
        }
    }

    public static final class TrainerConfig {
        public final float learningRate;
        public final float weightDecay;
        public final float maxGradNorm;
        public final int ignoreIndex;
        public final boolean useStreamingLmLoss;
        public final int gradientAccumulationSteps;

        public TrainerConfig(float learningRate, float weightDecay, float maxGradNorm, int ignoreIndex) {
            this(learningRate, weightDecay, maxGradNorm, ignoreIndex, true);
        }

        public TrainerConfig(
                float learningRate,
                float weightDecay,
                float maxGradNorm,
                int ignoreIndex,
                boolean useStreamingLmLoss
        ) {
            this(
                    learningRate,
                    weightDecay,
                    maxGradNorm,
                    ignoreIndex,
                    useStreamingLmLoss,
                    1
            );
        }

        public TrainerConfig(
                float learningRate,
                float weightDecay,
                float maxGradNorm,
                int ignoreIndex,
                boolean useStreamingLmLoss,
                int gradientAccumulationSteps
        ) {
            if (learningRate <= 0.0f) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            if (weightDecay < 0.0f) {
                throw new IllegalArgumentException("weightDecay must be non-negative");
            }
            if (maxGradNorm <= 0.0f) {
                throw new IllegalArgumentException("maxGradNorm must be positive");
            }
            if (gradientAccumulationSteps <= 0) {
                throw new IllegalArgumentException("gradientAccumulationSteps must be positive");
            }
            this.learningRate = learningRate;
            this.weightDecay = weightDecay;
            this.maxGradNorm = maxGradNorm;
            this.ignoreIndex = ignoreIndex;
            this.useStreamingLmLoss = useStreamingLmLoss;
            this.gradientAccumulationSteps = gradientAccumulationSteps;
        }

        public static TrainerConfig defaults() {
            return new TrainerConfig(2e-4f, 0.0f, 1.0f, -100);
        }
    }

    public static final class PreferenceStepResult {
        public final float loss;
        public final int trainableTensorCount;
        public final int pairCount;
        public final int validResponseTokenCount;
        public final float accumulatedLoss;
        public final float chosenReward;
        public final float rejectedReward;
        public final float rewardMargin;
        public final float rewardAccuracy;
        public final int accumulationStep;
        public final int gradientAccumulationSteps;
        public final boolean optimizerStep;

        private PreferenceStepResult(
                float loss,
                int trainableTensorCount,
                int pairCount,
                int validResponseTokenCount,
                float accumulatedLoss,
                float chosenReward,
                float rejectedReward,
                float rewardMargin,
                float rewardAccuracy,
                int accumulationStep,
                int gradientAccumulationSteps,
                boolean optimizerStep
        ) {
            this.loss = loss;
            this.trainableTensorCount = trainableTensorCount;
            this.pairCount = pairCount;
            this.validResponseTokenCount = validResponseTokenCount;
            this.accumulatedLoss = accumulatedLoss;
            this.chosenReward = chosenReward;
            this.rejectedReward = rejectedReward;
            this.rewardMargin = rewardMargin;
            this.rewardAccuracy = rewardAccuracy;
            this.accumulationStep = accumulationStep;
            this.gradientAccumulationSteps = gradientAccumulationSteps;
            this.optimizerStep = optimizerStep;
        }
    }

    public static final class PreferenceScoreResult {
        public final int validResponseTokenCount;
        public final float[] chosenLogps;
        public final float[] rejectedLogps;

        private PreferenceScoreResult(
                int validResponseTokenCount,
                float[] chosenLogps,
                float[] rejectedLogps
        ) {
            this.validResponseTokenCount = validResponseTokenCount;
            this.chosenLogps = Arrays.copyOf(chosenLogps, chosenLogps.length);
            this.rejectedLogps = Arrays.copyOf(rejectedLogps, rejectedLogps.length);
        }
    }

    public static final class KtoScoreResult {
        public final int validResponseTokenCount;
        public final int validKlResponseTokenCount;
        public final float[] responseLogps;
        public final float[] klResponseLogps;

        private KtoScoreResult(
                int validResponseTokenCount,
                int validKlResponseTokenCount,
                float[] responseLogps,
                float[] klResponseLogps
        ) {
            this.validResponseTokenCount = validResponseTokenCount;
            this.validKlResponseTokenCount = validKlResponseTokenCount;
            this.responseLogps = Arrays.copyOf(responseLogps, responseLogps.length);
            this.klResponseLogps = Arrays.copyOf(klResponseLogps, klResponseLogps.length);
        }
    }

    public static final class KtoStepResult {
        public final float loss;
        public final int trainableTensorCount;
        public final int pairCount;
        public final int sampleCount;
        public final int desirableCount;
        public final int undesirableCount;
        public final int validResponseTokenCount;
        public final int validKlResponseTokenCount;
        public final float accumulatedLoss;
        public final float desirableReward;
        public final float undesirableReward;
        public final float rewardMargin;
        public final float rewardAccuracy;
        public final float klEstimate;
        public final int accumulationStep;
        public final int gradientAccumulationSteps;
        public final boolean optimizerStep;

        private KtoStepResult(
                float loss,
                int trainableTensorCount,
                int pairCount,
                int sampleCount,
                int validResponseTokenCount,
                float accumulatedLoss,
                float desirableReward,
                float undesirableReward,
                float rewardMargin,
                float rewardAccuracy,
                int accumulationStep,
                int gradientAccumulationSteps,
                boolean optimizerStep,
                float klEstimate,
                int desirableCount,
                int undesirableCount,
                int validKlResponseTokenCount
        ) {
            this.loss = loss;
            this.trainableTensorCount = trainableTensorCount;
            this.pairCount = pairCount;
            this.sampleCount = sampleCount;
            this.desirableCount = desirableCount;
            this.undesirableCount = undesirableCount;
            this.validResponseTokenCount = validResponseTokenCount;
            this.validKlResponseTokenCount = validKlResponseTokenCount;
            this.accumulatedLoss = accumulatedLoss;
            this.desirableReward = desirableReward;
            this.undesirableReward = undesirableReward;
            this.rewardMargin = rewardMargin;
            this.rewardAccuracy = rewardAccuracy;
            this.klEstimate = klEstimate;
            this.accumulationStep = accumulationStep;
            this.gradientAccumulationSteps = gradientAccumulationSteps;
            this.optimizerStep = optimizerStep;
        }
    }

    public static final class TrainStepResult {
        public final float loss;
        public final int trainableTensorCount;
        public final int validLabelCount;
        public final float accumulatedLoss;
        public final int accumulationStep;
        public final int gradientAccumulationSteps;
        public final boolean optimizerStep;

        private TrainStepResult(
                float loss,
                int trainableTensorCount,
                int validLabelCount,
                float accumulatedLoss,
                int accumulationStep,
                int gradientAccumulationSteps,
                boolean optimizerStep
        ) {
            this.loss = loss;
            this.trainableTensorCount = trainableTensorCount;
            this.validLabelCount = validLabelCount;
            this.accumulatedLoss = accumulatedLoss;
            this.accumulationStep = accumulationStep;
            this.gradientAccumulationSteps = gradientAccumulationSteps;
            this.optimizerStep = optimizerStep;
        }
    }

    public static final class SelfTestResult {
        public final float loss;
        public final int trainableTensorCount;
        public final double elapsedMillis;

        private SelfTestResult(float loss, int trainableTensorCount, double elapsedMillis) {
            this.loss = loss;
            this.trainableTensorCount = trainableTensorCount;
            this.elapsedMillis = elapsedMillis;
        }
    }
}

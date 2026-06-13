package com.mobilefinetuner.visualizer.data

import com.google.common.truth.Truth.assertThat
import com.mobilefinetuner.visualizer.model.RunStatus
import org.junit.Test

class TrainingLogParserTest {

    @Test
    fun parseGpt2TrainLog_extractsMetricsAndSummary() {
        val lines = listOf(
            "[Train] epoch 1/0 | step 10/2251 (global 10/200) | lr 0.000100 | loss 4.0012 | ppl 54.66 | grad_norm 0.913 | tokens 1024",
            "[Train] epoch 1/0 | step 20/2251 (global 20/200) | lr 0.000200 | loss 3.6844 | ppl 39.82 | grad_norm 1.000 | tokens 1024",
            "[RSSSummary] MaxRSS(train) = 1822 MB",
            "\u2705 \u8bad\u7ec3\u5b8c\u6210\uff01",
            "  \u603b\u6b65\u6570: 200",
            "  \u603btokens: 204800",
            "  \u6700\u7ec8EMA loss: 3.1380"
        )

        val parsed = TrainingLogParser.parseTrainLog(lines)

        assertThat(parsed.status).isEqualTo(RunStatus.COMPLETED)
        assertThat(parsed.metrics).hasSize(2)
        assertThat(parsed.metrics.last().step).isEqualTo(20)
        assertThat(parsed.metrics.last().loss).isWithin(1e-6).of(3.6844)
        assertThat(parsed.summary.maxTrainRssMb).isWithin(1e-6).of(1822.0)
        assertThat(parsed.summary.finalEmaLoss).isWithin(1e-6).of(3.138)
        assertThat(parsed.summary.totalTokens).isEqualTo(204800)
        assertThat(parsed.summary.totalSteps).isEqualTo(200)
    }

    @Test
    fun parseGemmaTrainLog_extractsStepFormat() {
        val lines = listOf(
            "[Step 1] Loss=7.55168 PPL=1903.94 LR=7.14286e-06",
            "[Step 2] Loss=6.83055 PPL=925.699 LR=1.42857e-05",
            "[GemmaTrainer] Reached max_steps=30, stopping early.",
            "Gemma LoRA training completed"
        )

        val parsed = TrainingLogParser.parseTrainLog(lines)

        assertThat(parsed.metrics).hasSize(2)
        assertThat(parsed.metrics[0].step).isEqualTo(1)
        assertThat(parsed.metrics[0].ppl).isWithin(1e-6).of(1903.94)
        assertThat(parsed.metrics[1].lr).isWithin(1e-10).of(1.42857e-5)
        assertThat(parsed.status).isEqualTo(RunStatus.COMPLETED)
    }

    @Test
    fun parseRssCsv_extractsPoints() {
        val lines = listOf(
            "tick,rss_kb,rss_mb,timestamp",
            "0,2576,2,14:44:36",
            "1,542080,529,14:44:36",
            "2,399376,390,14:44:37"
        )

        val points = TrainingLogParser.parseRssCsv(lines)

        assertThat(points).hasSize(3)
        assertThat(points[1].index).isEqualTo(1)
        assertThat(points[1].rssMb).isWithin(1e-6).of(529.0)
        assertThat(points[1].timeLabel).isEqualTo("14:44:36")
    }

    @Test
    fun parseMetricsNdjson_extractsMetrics() {
        val lines = listOf(
            "{\"step\":1,\"loss\":4.2,\"ppl\":66.9,\"lr\":0.0002,\"tokens\":1024}",
            "{\"step\":2,\"loss\":4.0,\"ppl\":54.1,\"lr\":0.00019,\"tokens\":1024,\"reward_accuracy\":0.75,\"reward_margin\":0.42,\"chosen_reward\":0.2,\"rejected_reward\":-0.22,\"grad_norm\":1.0,\"rss_mb\":1840.5}"
        )

        val metrics = TrainingLogParser.parseMetricsNdjson(lines)

        assertThat(metrics).hasSize(2)
        assertThat(metrics[0].step).isEqualTo(1)
        assertThat(metrics[1].loss).isWithin(1e-6).of(4.0)
        assertThat(metrics[1].tokens).isEqualTo(1024)
        assertThat(metrics[1].rewardAccuracy).isWithin(1e-6).of(0.75)
        assertThat(metrics[1].rewardMargin).isWithin(1e-6).of(0.42)
        assertThat(metrics[1].chosenReward).isWithin(1e-6).of(0.2)
        assertThat(metrics[1].rejectedReward).isWithin(1e-6).of(-0.22)
        assertThat(metrics[1].gradNorm).isWithin(1e-6).of(1.0)
        assertThat(metrics[1].rssMb).isWithin(1e-6).of(1840.5)
    }

    @Test
    fun parseDpoTrainAndEvalLog_extractsHfrlMetrics() {
        val lines = listOf(
            "[Train] epoch 1/0 | step 1/16 (global 1/4) | lr 0.000500 | dpo_loss 0.6931 | grad_norm 1.000000 | nonzero_grad_tensors 24 | resp_tokens 18 | RSS(pre/fwd/bwd/opt/post)=100.0/150.0/170.0/171.0/169.0 MB | step_max=171.0 MB | peak=171.0 MB",
            "[Eval] epoch 1 | step 1 | loss 0.6862 | reward_acc 0.6200 | reward_margin 0.0149 | ema_loss 0.6931 | total_tokens 18"
        )

        val parsed = TrainingLogParser.parseTrainLog(lines)

        assertThat(parsed.metrics).hasSize(1)
        val metric = parsed.metrics.single()
        assertThat(metric.step).isEqualTo(1)
        assertThat(metric.totalSteps).isEqualTo(4)
        assertThat(metric.loss).isWithin(1e-6).of(0.6862)
        assertThat(metric.lr).isWithin(1e-10).of(0.0005)
        assertThat(metric.tokens).isEqualTo(18)
        assertThat(metric.gradNorm).isWithin(1e-6).of(1.0)
        assertThat(metric.rssMb).isWithin(1e-6).of(171.0)
        assertThat(metric.rewardAccuracy).isWithin(1e-6).of(0.62)
        assertThat(metric.rewardMargin).isWithin(1e-6).of(0.0149)
    }

    @Test
    fun parseNativeDemoStopMarker_returnsStoppedStatus() {
        val lines = listOf(
            "[1/6] Load pretrained model...",
            "[Stopped] User requested stop. Sending TERM to native pid 12345."
        )

        val parsed = TrainingLogParser.parseTrainLog(lines)

        assertThat(parsed.status).isEqualTo(RunStatus.STOPPED)
        assertThat(parsed.events.map { it.message }).contains(lines.last())
    }

    @Test
    fun parseNativeDemoStartupStage_returnsRunningBeforeFirstStep() {
        val lines = listOf(
            "[1/6] Load pretrained model...",
            "  total_steps    : 80",
            "[GPT2Model] Initialized with 12 layers, 768 hidden, 12 heads"
        )

        val parsed = TrainingLogParser.parseTrainLog(lines)

        assertThat(parsed.status).isEqualTo(RunStatus.RUNNING)
        assertThat(parsed.metrics).isEmpty()
        assertThat(parsed.summary.totalSteps).isEqualTo(80)
        assertThat(parsed.events.map { it.message }).contains(lines.first())
    }

    @Test
    fun parseInferenceSimCompleteMarker_returnsCompletedStatus() {
        val lines = listOf(
            "[Init] Qwen3 HFRL inference simulation",
            "[DPO] step=1/2 | loss=0.6300 | reward_acc=0.6600 | reward_margin=0.2400",
            "[Complete] inference_feedback_jsonl=runs/qwen3_hfrl_infer_sim/inference_feedback.jsonl"
        )

        val parsed = TrainingLogParser.parseTrainLog(lines)

        assertThat(parsed.status).isEqualTo(RunStatus.COMPLETED)
    }

    @Test
    fun parseInferenceJsonl_extractsGeneratedAndFeedbackRecords() {
        val generated = TrainingLogParser.parseGeneratedResponsesJsonl(
            listOf(
                "{\"prompt_id\":\"travel_fitness\",\"prompt\":\"How can I maintain fitness?\",\"completion\":\"Use hotel gyms\\nand walk.\",\"chosen\":\"Use hotel gyms and walk.\",\"elapsed_ms\":45296.1,\"source\":\"qwen3_native_inference\"}"
            )
        )
        val feedback = TrainingLogParser.parseInferenceFeedbackJsonl(
            listOf(
                "{\"prompt\":\"How can I maintain fitness?\",\"chosen\":\"Use hotel gyms and walk.\",\"rejected\":\"Ignore sleep.\",\"source\":\"qwen3_native_inference_sim\",\"prompt_id\":\"travel_fitness\"}"
            )
        )

        assertThat(generated).hasSize(1)
        assertThat(generated.single().promptId).isEqualTo("travel_fitness")
        assertThat(generated.single().completion).isEqualTo("Use hotel gyms\nand walk.")
        assertThat(generated.single().elapsedMs).isWithin(1e-6).of(45296.1)

        assertThat(feedback).hasSize(1)
        assertThat(feedback.single().chosen).isEqualTo("Use hotel gyms and walk.")
        assertThat(feedback.single().rejected).isEqualTo("Ignore sleep.")
    }

    @Test
    fun parseRealGpt2Log_sampleFileMatchesExpectedTailStep() {
        val lines = readResourceLines("samples/gpt2_train.log")
        val parsed = TrainingLogParser.parseTrainLog(lines)

        assertThat(parsed.metrics).isNotEmpty()
        assertThat(parsed.metrics.last().step).isEqualTo(200)
        assertThat(parsed.summary.totalSteps).isEqualTo(200)
        assertThat(parsed.summary.totalTokens).isEqualTo(204800)
        assertThat(parsed.status).isEqualTo(RunStatus.COMPLETED)
    }

    @Test
    fun parseRealGemmaLog_sampleFileMatchesExpectedTailStep() {
        val lines = readResourceLines("samples/gemma_train.log")
        val parsed = TrainingLogParser.parseTrainLog(lines)

        assertThat(parsed.metrics).isNotEmpty()
        assertThat(parsed.metrics.last().step).isEqualTo(30)
        assertThat(parsed.status).isEqualTo(RunStatus.COMPLETED)
    }

    @Test
    fun parseRealRssCsv_sampleFileParsesManyPoints() {
        val lines = readResourceLines("samples/rss.csv")
        val points = TrainingLogParser.parseRssCsv(lines)

        assertThat(points.size).isAtLeast(50)
        assertThat(points.first().index).isEqualTo(0)
        assertThat(points.first().rssMb).isWithin(1e-6).of(2.0)
    }

    private fun readResourceLines(path: String): List<String> {
        val stream = javaClass.classLoader?.getResourceAsStream(path)
        checkNotNull(stream) { "Missing test resource: $path" }
        return stream.bufferedReader().readLines()
    }
}

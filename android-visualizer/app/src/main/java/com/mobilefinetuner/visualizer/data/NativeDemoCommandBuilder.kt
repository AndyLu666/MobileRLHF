package com.mobilefinetuner.visualizer.data

import com.mobilefinetuner.visualizer.model.NativeDemoConfig

object NativeDemoCommandBuilder {

    data class Paths(
        val binaryPath: String = "/data/local/tmp/mft_train",
        val pretrainedDir: String = "/data/local/tmp/mft_gpt2",
        val trainJsonl: String,
        val evalJsonl: String,
        val runDir: String,
        val loraOut: String = "adapter.safetensors",
        val metricsOut: String = "metrics.ndjson",
        val evalOut: String = "eval_metrics.jsonl",
        val trainLog: String = "train.log",
        val pidFile: String = "native.pid"
    )

    fun buildShellCommand(config: NativeDemoConfig, paths: Paths): String {
        val args = listOf(
            paths.binaryPath,
            "--train_mode", "dpo",
            "--pretrained_dir", paths.pretrainedDir,
            "--preference_train", paths.trainJsonl,
            "--preference_valid", paths.evalJsonl,
            "--lora_out", paths.loraOut,
            "--metrics_out", paths.metricsOut,
            "--eval_out", paths.evalOut,
            "--steps", config.steps.toString(),
            "--batch_size", "1",
            "--grad_accum_steps", "1",
            "--seq_len", config.seqLen.toString(),
            "--rank", config.rank.toString(),
            "--alpha", config.alpha.toString(),
            "--lr", config.learningRate.toString(),
            "--log_interval", "1",
            "--eval_interval", config.evalInterval.toString(),
            "--eval_batches", config.evalBatches.toString(),
            "--eval_batch_size", "1",
            "--dpo_pair_forward", if (config.dpoPairForward) "1" else "0",
            "--cache_ref_logps", if (config.cacheRefLogps) "1" else "0",
            "--seed", "42"
        ).joinToString(" ") { shellQuote(it) }

        val quotedLog = shellQuote(paths.trainLog)
        val quotedPid = shellQuote(paths.pidFile)
        return "cd ${shellQuote(paths.runDir)} && " +
            "rm -f $quotedPid && " +
            "($args > $quotedLog 2>&1 & echo \$! > $quotedPid; wait \$(cat $quotedPid); exit \$?)"
    }

    fun shellQuote(raw: String): String {
        if (raw.matches(Regex("[A-Za-z0-9_./:=+-]+"))) return raw
        return "'" + raw.replace("'", "'\"'\"'") + "'"
    }
}

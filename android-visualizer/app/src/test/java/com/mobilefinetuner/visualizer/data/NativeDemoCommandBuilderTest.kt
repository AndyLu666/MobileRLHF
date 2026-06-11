package com.mobilefinetuner.visualizer.data

import com.google.common.truth.Truth.assertThat
import com.mobilefinetuner.visualizer.model.NativeDemoConfig
import org.junit.Test

class NativeDemoCommandBuilderTest {

    @Test
    fun buildShellCommand_includesCoreDpoArguments() {
        val command = NativeDemoCommandBuilder.buildShellCommand(
            config = NativeDemoConfig(
                steps = 80,
                seqLen = 128,
                rank = 1,
                alpha = 2,
                learningRate = 0.001,
                evalInterval = 20,
                cacheRefLogps = false
            ),
            paths = NativeDemoCommandBuilder.Paths(
                trainJsonl = "/sdcard/train.jsonl",
                evalJsonl = "/sdcard/eval.jsonl",
                runDir = "/sdcard/run"
            )
        )

        assertThat(command).contains("cd /sdcard/run")
        assertThat(command).contains("/data/local/tmp/mft_train")
        assertThat(command).contains("--train_mode dpo")
        assertThat(command).contains("--steps 80")
        assertThat(command).contains("--seq_len 128")
        assertThat(command).contains("--rank 1")
        assertThat(command).contains("--alpha 2")
        assertThat(command).contains("--eval_interval 20")
        assertThat(command).contains("--cache_ref_logps 0")
        assertThat(command).contains("> train.log 2>&1")
        assertThat(command).contains("echo \$! > native.pid")
        assertThat(command).contains("wait \$(cat native.pid)")
    }

    @Test
    fun buildShellCommand_quotesPathsWithSpaces() {
        val command = NativeDemoCommandBuilder.buildShellCommand(
            config = NativeDemoConfig(steps = 8, cacheRefLogps = true),
            paths = NativeDemoCommandBuilder.Paths(
                trainJsonl = "/sdcard/My Runs/train.jsonl",
                evalJsonl = "/sdcard/My Runs/eval.jsonl",
                runDir = "/sdcard/My Runs/out"
            )
        )

        assertThat(command).contains("cd '/sdcard/My Runs/out'")
        assertThat(command).contains("--preference_train '/sdcard/My Runs/train.jsonl'")
        assertThat(command).contains("--preference_valid '/sdcard/My Runs/eval.jsonl'")
        assertThat(command).contains("--cache_ref_logps 1")
    }
}

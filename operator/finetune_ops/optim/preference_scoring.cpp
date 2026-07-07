#include "preference_scoring.h"

#include "../core/dpo_loss.h"

#include <stdexcept>

namespace ops {

PreferenceLogps score_preference_logps(AutoModelForCausalLM& model,
                                       const PreferenceBatch& batch) {
    if (batch.batch_size <= 0) {
        throw std::runtime_error("score_preference_logps: empty preference batch");
    }
    if (!batch.chosen_input_ids || !batch.rejected_input_ids ||
        !batch.chosen_attention_mask || !batch.rejected_attention_mask ||
        !batch.chosen_response_mask || !batch.rejected_response_mask) {
        throw std::runtime_error("score_preference_logps: incomplete preference batch tensors");
    }

    const TensorPtr lm_head = model.lm_head_weight_for_loss();
    const TensorPtr chosen_hidden =
        model.forward_hidden(batch.chosen_input_ids, batch.chosen_attention_mask);
    const TensorPtr rejected_hidden =
        model.forward_hidden(batch.rejected_input_ids, batch.rejected_attention_mask);

    PreferenceLogps scores;
    scores.chosen_logps = sequence_logprob_masked_from_hidden(
        chosen_hidden, lm_head, batch.chosen_input_ids, batch.chosen_response_mask);
    scores.rejected_logps = sequence_logprob_masked_from_hidden(
        rejected_hidden, lm_head, batch.rejected_input_ids, batch.rejected_response_mask);
    scores.valid_response_token_count = batch.valid_response_token_count;
    return scores;
}

}  // namespace ops

#pragma once

#include "../data/kto_batch.h"
#include "../data/preference_batch.h"
#include "../graph/auto_model.h"

#include <vector>

namespace ops {

struct PreferenceLogps {
    std::vector<float> chosen_logps;
    std::vector<float> rejected_logps;
    int valid_response_token_count = 0;
};

PreferenceLogps score_preference_logps(AutoModelForCausalLM& model,
                                       const PreferenceBatch& batch);

struct KTOLogps {
    std::vector<float> response_logps;
    std::vector<float> kl_response_logps;
    int valid_response_token_count = 0;
    int valid_kl_response_token_count = 0;
};

/** Score both the matched and rotated-completion branches of a KTO batch. */
KTOLogps score_kto_logps(AutoModelForCausalLM& model,
                         const KTOBatch& batch);

}  // namespace ops

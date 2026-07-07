#pragma once

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

}  // namespace ops

#include "dpo_loss.h"
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ops;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(float actual, float expected, float tol, const std::string& name) {
    if (std::fabs(actual - expected) > tol) {
        throw std::runtime_error(name + " mismatch actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

float max_abs_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* ad = a->data<float>();
    const float* bd = b->data<float>();
    float max_diff = 0.0f;
    for (int64_t i = 0; i < a->numel(); ++i) {
        max_diff = std::max(max_diff, std::fabs(ad[i] - bd[i]));
    }
    return max_diff;
}

TensorPtr make_i32(const std::vector<int64_t>& shape, const std::vector<int32_t>& values) {
    return std::make_shared<Tensor>(shape, values.data(), kInt32, kCPU);
}

void test_metrics_from_logps() {
    const std::vector<float> pc{-1.0f, -2.0f};
    const std::vector<float> pr{-2.0f, -1.0f};
    const std::vector<float> rc{-1.5f, -2.5f};
    const std::vector<float> rr{-2.5f, -1.5f};
    const float beta = 0.1f;
    const DPOMetrics m = compute_dpo_metrics_from_logps(pc, pr, rc, rr, beta);

    const double margin0 = beta * ((pc[0] - pr[0]) - (rc[0] - rr[0]));
    const double margin1 = beta * ((pc[1] - pr[1]) - (rc[1] - rr[1]));
    const double expected =
        (std::log1p(std::exp(-margin0)) + std::log1p(std::exp(-margin1))) / 2.0;
    require_close(m.loss, static_cast<float>(expected), 1e-6f, "DPO metrics loss");
    require_close(m.reward_accuracy, 0.0f, 1e-6f, "DPO reward accuracy");
}

void test_kto_metrics_from_logps() {
    const std::vector<float> policy{-1.0f, -2.0f};
    const std::vector<float> reference{-1.5f, -1.0f};
    const std::vector<float> labels{1.0f, -1.0f};
    const float beta = 0.2f;
    const DPOMetrics m = compute_kto_metrics_from_logps(
        policy, reference, labels, beta, 1.0f, 1.0f);

    const double logratio0 = policy[0] - reference[0];
    const double logratio1 = policy[1] - reference[1];
    const double kl_center = std::max((logratio0 + logratio1) / 2.0, 0.0);
    const double reward0 = beta * (logratio0 - kl_center);
    const double reward1 = beta * (logratio1 - kl_center);
    const double expected_loss =
        ((1.0 - (1.0 / (1.0 + std::exp(-reward0)))) +
         (1.0 - (1.0 / (1.0 + std::exp(reward1))))) / 2.0;

    require_close(m.loss, static_cast<float>(expected_loss), 1e-6f, "KTO metrics loss");
    require_close(m.chosen_reward, static_cast<float>(reward0), 1e-6f, "KTO desirable reward");
    require_close(m.rejected_reward, static_cast<float>(reward1), 1e-6f, "KTO undesirable reward");
    require_close(m.reward_margin, static_cast<float>(reward0 - reward1), 1e-6f, "KTO reward margin");
    require_close(m.reward_accuracy, 1.0f, 1e-6f, "KTO reward accuracy");
}

void test_canonical_kto_fixed_numeric_and_gradient_fixture() {
    const std::vector<float> policy{-1.0f, -2.0f, -0.5f, -1.5f};
    const std::vector<float> reference{-1.2f, -1.8f, -0.7f, -1.4f};
    const std::vector<float> policy_kl{-1.1f, -1.9f, -0.8f, -1.3f};
    const std::vector<float> reference_kl{-1.3f, -1.7f, -0.9f, -1.4f};
    const std::vector<float> labels{1.0f, 1.0f, -1.0f, -1.0f};
    constexpr float beta = 0.1f;
    constexpr float desirable_weight = 1.5f;
    constexpr float undesirable_weight = 0.75f;

    const DPOMetrics metrics = compute_kto_metrics_from_logps_and_kl(
        policy, policy_kl, reference, reference_kl, labels, beta,
        desirable_weight, undesirable_weight);
    require_close(metrics.kl_estimate, 0.05f, 1e-6f,
                  "canonical KTO KL estimate");
    require_close(metrics.loss, 0.5634374043f, 1e-6f,
                  "canonical KTO fixed loss");
    require_close(metrics.chosen_reward, 0.0f, 1e-6f,
                  "canonical KTO desirable reward");
    require_close(metrics.rejected_reward, 0.005f, 1e-6f,
                  "canonical KTO undesirable reward");
    require_close(metrics.reward_margin, -0.005f, 1e-6f,
                  "canonical KTO reward margin");
    require_close(metrics.reward_accuracy, 0.5f, 1e-6f,
                  "canonical KTO reward accuracy");

    const int64_t B = 4;
    const int64_t S = 2;
    const int64_t V = 2;
    auto logits = zeros({B, S, V}, kFloat32, kCPU);
    auto kl_logits = zeros({B, S, V}, kFloat32, kCPU);
    logits->set_requires_grad(true);
    kl_logits->set_requires_grad(true);
    auto ids = make_i32({B, S}, {0, 1, 0, 1, 0, 1, 0, 1});
    auto response_mask = make_i32({B, S}, {0, 1, 0, 1, 0, 1, 0, 1});

    const float base_logp = -std::log(2.0f);
    const std::vector<float> ratios{0.2f, -0.2f, 0.2f, -0.1f};
    const std::vector<float> kl_ratios{0.2f, -0.2f, 0.1f, 0.1f};
    std::vector<float> dense_reference;
    std::vector<float> dense_reference_kl;
    for (std::size_t i = 0; i < ratios.size(); ++i) {
        dense_reference.push_back(base_logp - ratios[i]);
        dense_reference_kl.push_back(base_logp - kl_ratios[i]);
    }

    auto loss = kto_loss_with_ref_logps_and_kl(
        logits, kl_logits, dense_reference, dense_reference_kl,
        ids, ids, response_mask, response_mask, labels, beta,
        desirable_weight, undesirable_weight);
    require_close(loss->data<float>()[0], 0.5634374043f, 1e-6f,
                  "canonical KTO dense fixed loss");
    loss->backward();
    require(logits->grad() != nullptr, "canonical KTO missing matched logits grad");
    require(kl_logits->grad() == nullptr,
            "canonical KTO must detach the mismatched KL branch");

    const float expected_coeffs[] = {
        -0.009374472676f,
        -0.009373535309f,
         0.004687236338f,
         0.004687236338f,
    };
    const float* grad = logits->grad()->data<float>();
    for (int64_t b = 0; b < B; ++b) {
        const int64_t row = (b * S) * V;
        require_close(grad[row], -0.5f * expected_coeffs[b], 1e-6f,
                      "canonical KTO non-target logit gradient");
        require_close(grad[row + 1], 0.5f * expected_coeffs[b], 1e-6f,
                      "canonical KTO target logit gradient");
        require_close(grad[row + V], 0.0f, 1e-7f,
                      "canonical KTO unused timestep gradient");
        require_close(grad[row + V + 1], 0.0f, 1e-7f,
                      "canonical KTO unused target gradient");
    }
}

void test_dense_backward_has_policy_grads() {
    auto A = zeros({2, 2}, kFloat32, kCPU);
    auto B = zeros({2, 3}, kFloat32, kCPU);
    A->set_requires_grad(true);
    B->set_requires_grad(true);
    {
        float* a = A->data<float>();
        float* b = B->data<float>();
        a[0] = 0.4f; a[1] = -0.2f; a[2] = 0.1f; a[3] = 0.3f;
        b[0] = 0.2f; b[1] = -0.1f; b[2] = 0.5f;
        b[3] = -0.3f; b[4] = 0.4f; b[5] = 0.1f;
    }

    auto x_chosen = zeros({1, 2, 2}, kFloat32, kCPU);
    auto x_rejected = zeros({1, 2, 2}, kFloat32, kCPU);
    {
        float* xc = x_chosen->data<float>();
        float* xr = x_rejected->data<float>();
        xc[0] = 1.0f; xc[1] = 0.5f; xc[2] = -0.2f; xc[3] = 0.7f;
        xr[0] = 1.0f; xr[1] = 0.5f; xr[2] = 0.3f; xr[3] = -0.4f;
    }

    auto policy_chosen_logits = matmul(matmul(x_chosen, A), B);
    auto policy_rejected_logits = matmul(matmul(x_rejected, A), B);
    auto ref_chosen_logits = policy_chosen_logits->detach();
    auto ref_rejected_logits = policy_rejected_logits->detach();

    auto chosen_ids = make_i32({1, 2}, {0, 1});
    auto rejected_ids = make_i32({1, 2}, {0, 2});
    auto response_mask = make_i32({1, 2}, {0, 1});

    auto loss = dpo_loss(policy_chosen_logits, policy_rejected_logits,
                         ref_chosen_logits, ref_rejected_logits,
                         chosen_ids, rejected_ids,
                         response_mask, response_mask,
                         0.1f);
    loss->backward();

    require(std::isfinite(loss->data<float>()[0]), "dense DPO loss is not finite");
    require(A->grad() != nullptr, "dense DPO missing A grad");
    require(B->grad() != nullptr, "dense DPO missing B grad");
    require(max_abs_diff(A->grad(), zeros(A->shape(), kFloat32, kCPU)) > 0.0f,
            "dense DPO A grad is zero");
    require(max_abs_diff(B->grad(), zeros(B->shape(), kFloat32, kCPU)) > 0.0f,
            "dense DPO B grad is zero");
}

void test_streaming_matches_dense_loss_and_grad() {
    const int64_t Bsz = 2;
    const int64_t S = 3;
    const int64_t H = 2;
    const int64_t V = 4;
    std::vector<float> chosen_values = {
        0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.2f,
        -0.1f, 0.2f, 0.0f, 0.3f, 0.4f, -0.2f,
    };
    std::vector<float> rejected_values = {
        -0.2f, 0.1f, 0.2f, -0.3f, 0.5f, -0.4f,
        0.3f, 0.2f, -0.1f, 0.4f, -0.3f, 0.1f,
    };
    std::vector<float> weight_values = {
        0.2f, -0.1f,
        -0.3f, 0.4f,
        0.1f, 0.2f,
        -0.2f, 0.3f,
    };
    auto chosen_ids = make_i32({Bsz, S}, {0, 1, 2, 0, 2, 3});
    auto rejected_ids = make_i32({Bsz, S}, {0, 2, 1, 0, 1, 3});
    auto response_mask = make_i32({Bsz, S}, {0, 1, 1, 0, 1, 1});

    auto chosen_dense = std::make_shared<Tensor>(std::vector<int64_t>{Bsz, S, H}, chosen_values.data(), kFloat32, kCPU);
    auto rejected_dense = std::make_shared<Tensor>(std::vector<int64_t>{Bsz, S, H}, rejected_values.data(), kFloat32, kCPU);
    auto weight_dense = std::make_shared<Tensor>(std::vector<int64_t>{V, H}, weight_values.data(), kFloat32, kCPU);
    chosen_dense->set_requires_grad(true);
    rejected_dense->set_requires_grad(true);
    weight_dense->set_requires_grad(true);
    auto dense_chosen_logits = matmul_rhs_T(chosen_dense, weight_dense);
    auto dense_rejected_logits = matmul_rhs_T(rejected_dense, weight_dense);
    std::vector<float> ref_chosen{0.0f, -0.1f};
    std::vector<float> ref_rejected{-0.2f, 0.1f};
    auto dense_loss = dpo_loss_with_ref_logps(dense_chosen_logits, dense_rejected_logits,
                                              ref_chosen, ref_rejected,
                                              chosen_ids, rejected_ids,
                                              response_mask, response_mask,
                                              0.2f);
    dense_loss->backward();

    auto chosen_stream = std::make_shared<Tensor>(std::vector<int64_t>{Bsz, S, H}, chosen_values.data(), kFloat32, kCPU);
    auto rejected_stream = std::make_shared<Tensor>(std::vector<int64_t>{Bsz, S, H}, rejected_values.data(), kFloat32, kCPU);
    auto weight_stream = std::make_shared<Tensor>(std::vector<int64_t>{V, H}, weight_values.data(), kFloat32, kCPU);
    chosen_stream->set_requires_grad(true);
    rejected_stream->set_requires_grad(true);
    weight_stream->set_requires_grad(true);
    auto streaming_loss = streaming_dpo_loss_with_ref_logps(chosen_stream, rejected_stream, weight_stream,
                                                            ref_chosen, ref_rejected,
                                                            chosen_ids, rejected_ids,
                                                            response_mask, response_mask,
                                                            0.2f);
    streaming_loss->backward();

    require_close(streaming_loss->data<float>()[0], dense_loss->data<float>()[0],
                  1e-5f, "streaming DPO loss");
    require(max_abs_diff(chosen_stream->grad(), chosen_dense->grad()) < 1e-5f,
            "streaming chosen hidden grad mismatch");
    require(max_abs_diff(rejected_stream->grad(), rejected_dense->grad()) < 1e-5f,
            "streaming rejected hidden grad mismatch");
    require(max_abs_diff(weight_stream->grad(), weight_dense->grad()) < 1e-5f,
            "streaming weight grad mismatch");
}

void test_streaming_kto_matches_dense_loss_and_grad() {
    const int64_t Bsz = 2;
    const int64_t S = 3;
    const int64_t H = 2;
    const int64_t V = 4;
    std::vector<float> values = {
        0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.2f,
        -0.1f, 0.2f, 0.0f, 0.3f, 0.4f, -0.2f,
    };
    std::vector<float> weight_values = {
        0.2f, -0.1f,
        -0.3f, 0.4f,
        0.1f, 0.2f,
        -0.2f, 0.3f,
    };
    auto ids = make_i32({Bsz, S}, {0, 1, 2, 0, 2, 3});
    auto response_mask = make_i32({Bsz, S}, {0, 1, 1, 0, 1, 1});
    const std::vector<float> ref_logps{0.0f, -0.1f};
    const std::vector<float> labels{1.0f, -1.0f};

    auto dense_hidden = std::make_shared<Tensor>(std::vector<int64_t>{Bsz, S, H}, values.data(), kFloat32, kCPU);
    auto dense_weight = std::make_shared<Tensor>(std::vector<int64_t>{V, H}, weight_values.data(), kFloat32, kCPU);
    dense_hidden->set_requires_grad(true);
    dense_weight->set_requires_grad(true);
    auto dense_logits = matmul_rhs_T(dense_hidden, dense_weight);
    auto dense_loss = kto_loss_with_ref_logps(
        dense_logits, ref_logps, ids, response_mask, labels, 0.2f, 1.0f, 1.0f);
    dense_loss->backward();

    auto stream_hidden = std::make_shared<Tensor>(std::vector<int64_t>{Bsz, S, H}, values.data(), kFloat32, kCPU);
    auto stream_weight = std::make_shared<Tensor>(std::vector<int64_t>{V, H}, weight_values.data(), kFloat32, kCPU);
    stream_hidden->set_requires_grad(true);
    stream_weight->set_requires_grad(true);
    auto streaming_loss = streaming_kto_loss_with_ref_logps(
        stream_hidden, stream_weight, ref_logps, ids, response_mask, labels, 0.2f, 1.0f, 1.0f);
    streaming_loss->backward();

    require_close(streaming_loss->data<float>()[0], dense_loss->data<float>()[0],
                  1e-5f, "streaming KTO loss");
    require(max_abs_diff(stream_hidden->grad(), dense_hidden->grad()) < 1e-5f,
            "streaming KTO hidden grad mismatch");
    require(max_abs_diff(stream_weight->grad(), dense_weight->grad()) < 1e-5f,
            "streaming KTO weight grad mismatch");
}

}  // namespace

int main() {
    test_metrics_from_logps();
    test_kto_metrics_from_logps();
    test_canonical_kto_fixed_numeric_and_gradient_fixture();
    test_dense_backward_has_policy_grads();
    test_streaming_matches_dense_loss_and_grad();
    test_streaming_kto_matches_dense_loss_and_grad();
    std::cout << "[PASS] DPO loss tests passed\n";
    return 0;
}

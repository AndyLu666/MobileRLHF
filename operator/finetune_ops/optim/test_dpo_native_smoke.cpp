#include "../core/dpo_loss.h"
#include "../core/ops.h"
#include <cmath>
#include <iostream>

using namespace ops;

int main() {
    auto A = zeros({2, 2}, kFloat32, kCPU);
    auto B = zeros({2, 3}, kFloat32, kCPU);
    A->set_requires_grad(true);
    B->set_requires_grad(true);

    {
        float* a = A->data<float>();
        float* b = B->data<float>();
        a[0] = 0.4f; a[1] = -0.2f;
        a[2] = 0.1f; a[3] = 0.3f;
        b[0] = 0.2f; b[1] = -0.1f; b[2] = 0.5f;
        b[3] = -0.3f; b[4] = 0.4f; b[5] = 0.1f;
    }

    auto x_chosen = zeros({1, 2, 2}, kFloat32, kCPU);
    auto x_rejected = zeros({1, 2, 2}, kFloat32, kCPU);
    {
        float* xc = x_chosen->data<float>();
        float* xr = x_rejected->data<float>();
        xc[0] = 1.0f; xc[1] = 0.5f;
        xc[2] = -0.2f; xc[3] = 0.7f;
        xr[0] = 1.0f; xr[1] = 0.5f;
        xr[2] = 0.3f; xr[3] = -0.4f;
    }

    auto hidden_chosen = matmul(x_chosen, A);
    auto hidden_rejected = matmul(x_rejected, A);
    auto policy_chosen_logits = matmul(hidden_chosen, B);
    auto policy_rejected_logits = matmul(hidden_rejected, B);

    auto ref_chosen_logits = policy_chosen_logits->detach();
    auto ref_rejected_logits = policy_rejected_logits->detach();

    auto chosen_ids = zeros({1, 2}, kInt32, kCPU);
    auto rejected_ids = zeros({1, 2}, kInt32, kCPU);
    auto response_mask = zeros({1, 2}, kInt32, kCPU);
    chosen_ids->data<int32_t>()[0] = 0;
    chosen_ids->data<int32_t>()[1] = 1;
    rejected_ids->data<int32_t>()[0] = 0;
    rejected_ids->data<int32_t>()[1] = 2;
    response_mask->data<int32_t>()[0] = 0;
    response_mask->data<int32_t>()[1] = 1;

    auto loss = dpo_loss(
        policy_chosen_logits,
        policy_rejected_logits,
        ref_chosen_logits,
        ref_rejected_logits,
        chosen_ids,
        rejected_ids,
        response_mask,
        response_mask,
        0.1f);
    loss->backward();

    auto grad_A = A->grad();
    auto grad_B = B->grad();
    double max_abs_A = 0.0;
    double max_abs_B = 0.0;
    if (grad_A) {
        const float* g = grad_A->data<float>();
        for (int64_t i = 0; i < grad_A->numel(); ++i) {
            max_abs_A = std::max(max_abs_A, std::fabs(static_cast<double>(g[i])));
        }
    }
    if (grad_B) {
        const float* g = grad_B->data<float>();
        for (int64_t i = 0; i < grad_B->numel(); ++i) {
            max_abs_B = std::max(max_abs_B, std::fabs(static_cast<double>(g[i])));
        }
    }

    std::cout << "loss=" << loss->item<float>()
              << " grad_A_max=" << max_abs_A
              << " grad_B_max=" << max_abs_B
              << std::endl;

    if (!grad_A || !grad_B || max_abs_A <= 0.0 || max_abs_B <= 0.0) {
        std::cerr << "DPO native smoke failed: expected non-zero grads on A and B" << std::endl;
        return 1;
    }
    return 0;
}

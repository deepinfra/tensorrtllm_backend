/* structured_logit_processor.cu
 * Copyright 2024 DeepInfra, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace triton::backend::inflight_batcher_llm {
using tensorrt_llm::runtime::SizeType32;

template <typename T> __global__ void applyLogitMask(
        T* const* logits, SizeType32 vocabSize, std::int32_t *states, std::int32_t *endIds, SizeType32 batchSize,
        const bool *maskVector, SizeType32 maskTokenSizePadded, SizeType32 extraVocabSize)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    T *batchLogits = logits[batchIdx];
    std::int32_t state = states[batchIdx];
    int endId = endIds[batchIdx];
    if (state < 0) {
        return;
    }
    const T negInf = static_cast<T>(-INFINITY);
    for (auto index = maskTokenSizePadded + static_cast<SizeType32>(threadIdx.x);
        index < extraVocabSize; index += static_cast<SizeType32>(blockDim.x))
    {
        if (state != 0 || index != endId) {
            batchLogits[index] = negInf;
        }
    }
    if (state == 0 && endId >= 0 && endId < vocabSize) {
        // Special case for end state to force a stop.
        for (auto index = static_cast<SizeType32>(threadIdx.x); index < vocabSize;
             index += static_cast<SizeType32>(blockDim.x))
        {
            if (index != endId) {
                batchLogits[index] = negInf;
            }
        }
    } else {
        const bool *stateMaskVector = maskVector + maskTokenSizePadded * state;
        for (auto index = static_cast<SizeType32>(threadIdx.x); index < vocabSize;
             index += static_cast<SizeType32>(blockDim.x))
        {
            if (!stateMaskVector[index]) {
                batchLogits[index] = negInf;
            }
        }
    }
}

template <typename T>
void invokeApplyLogitMask(
        T* const* logits, SizeType32 vocabSize, std::int32_t *states, std::int32_t *endIds, SizeType32 batchSize,
        const bool *maskVector, SizeType32 maskTokenSizePadded, cudaStream_t stream)
{
    dim3 block (512);
    dim3 grid (batchSize);
    SizeType32 extraVocabSize = vocabSize;
    if (maskTokenSizePadded < vocabSize) {
        vocabSize = maskTokenSizePadded;
    }

    applyLogitMask<<<grid, block, 0, stream>>>(logits,
        vocabSize, states, endIds, batchSize,
        maskVector, maskTokenSizePadded, extraVocabSize);
    sync_check_cuda_error(stream);
}

template void invokeApplyLogitMask(
    float* const* logits, SizeType32 logits_token_size, std::int32_t *states, std::int32_t *endIds, SizeType32 batch_size,
    const bool *mask_vector, SizeType32 mask_token_size_padded, cudaStream_t stream);
template void invokeApplyLogitMask(
    half* const* logits, SizeType32 logits_token_size, std::int32_t *states, std::int32_t *endIds, SizeType32 batch_size,
    const bool *mask_vector, SizeType32 mask_token_size_padded, cudaStream_t stream);
template void invokeApplyLogitMask(
    nv_bfloat16* const* logits, SizeType32 logits_token_size, std::int32_t *states, std::int32_t *endIds, SizeType32 batch_size,
    const bool *mask_vector, SizeType32 mask_token_size_padded, cudaStream_t stream);

}

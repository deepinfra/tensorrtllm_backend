/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/llmRequest.h" // TODO forward declare
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#define private protected
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#undef private

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class KVCacheBlock;

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using BlockPtr = std::shared_ptr<KVCacheBlock>;
using FreeBlocksQueue = std::list<BlockPtr>;
using NextBlockMap = std::unordered_map<VecTokens, BlockPtr>;

class KVCacheWrapper : public KVCacheManager
{
public:

    KVCacheWrapper(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, bool useOneMoreBlock,
        CudaStreamPtr stream, bool enableBlockReuse = false, bool onboardBlocks = true,
        CacheType cacheType = CacheType::kSELF);

    void allocatePools(nvinfer1::DataType dtype, bool useUvm = false);

    void startScheduling();

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request by one or two
    /// iterations
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const;

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request to completion (i.e. for
    /// maxNewTokens)
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getNeededBlocksToCompletion(LlmRequest const& req) const;

    void addContextTokens(SizeType32 seqSlotIdx, SizeType32 numTokens);

    /// @brief Increase size for request at seqSlotIdx. Allocate new KV cache block(s) if needed.
    void addToken(SizeType32 seqSlotIdx);

    /// @brief Add new request to the KV cache manager.
    /// @param inputLength Input length for which KV cache need to be allocated.
    /// @param beamWidth Beam width for which KV cache need to be allocated.
    /// @param llmRequest Optional request to use for KV cache lookup.
    /// @details If llmRequest is supplied and KV cache reuse is enabled, try to recover KV cache blocks for
    /// inputLength - 1 tokens and populate prepopulatedPromptLen.
    void addSequence(SizeType32 seqSlotIdx, SizeType32 inputLength, SizeType32 beamWidth,
        std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

    void removeSequence(SizeType32 seqSlotIdx, std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

    void schedulingRemoveSequence(SizeType32 seqSlotIdx);

    [[nodiscard]] runtime::ITensor::UniquePtr getBlockPoolPointers() const;

    void getBlockOffsetsOfBatch(
        runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const;

    //! @return maxBlockCount of all beams
    SizeType32 copyBlockOffsets(
        runtime::ITensor& output, SizeType32 outputSlotOffset, SizeType32 seqSlotIdx, SizeType32 beamWidth) const;

    // Volume of [2, numKvHeads, tokensPerBlock, sizePerHead]
    [[nodiscard]] static SizeType32 constexpr calculatePageSize(tensorrt_llm::runtime::ModelConfig const& modelConfig)
    {
        return 2 * modelConfig.getNbKvHeads() * modelConfig.getTokensPerBlock() * modelConfig.getSizePerHead();
    }

    // numLayers * 2 * numKvHeads * sizePerHead
    [[nodiscard]] static SizeType32 constexpr calculateCacheSizePerToken(
        tensorrt_llm::runtime::ModelConfig const& modelConfig, tensorrt_llm::runtime::WorldConfig const& worldConfig)
    {
        return modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism()) * 2 * modelConfig.getNbKvHeads()
            * modelConfig.getSizePerHead();
    }

    [[nodiscard]] static std::tuple<SizeType32, SizeType32> const calculateMaxNumBlocks(KvCacheConfig const& config,
        nvinfer1::DataType dtype, tensorrt_llm::runtime::ModelConfig const& modelConfig,
        tensorrt_llm::runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager);

    [[nodiscard]] bool isEnableBlockReuse() const
    {
        return mEnableBlockReuse;
    }

    void removeToken(SizeType32 seqSlotIdx);
    void rewindKVCache(SizeType32 seqSlotIdx, SizeType32 rewindLengths);

    [[nodiscard]] GenerationRequest const& getSequence(SizeType32 seqSlotIdx) const;

    [[nodiscard]] bool isCrossKv() const
    {
        return mCacheType == CacheType::kCROSS;
    }

    [[nodiscard]] static SizeType32 getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock);

    [[nodiscard]] static SizeType32 getMaxAttentionWindowUpperBound(SizeType32 blocksInPrimaryPool,
        SizeType32 tokensPerBlock, SizeType32 maxBeamWidth, SizeType32 sinkTokenLen, bool useOneMoreBlock);
};

KVCacheWrapper::KVCacheWrapper(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
    SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, bool useOneMoreBlock,
    CudaStreamPtr stream, bool enableBlockReuse, bool onboardBlocks, CacheType cacheType)
    : KVCacheManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool, blocksInSecondaryPool, maxNumSequences,
            maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream, enableBlockReuse, onboardBlocks, cacheType) {
}

void KVCacheWrapper::allocatePools(nvinfer1::DataType dtype, bool useUvm) {
    TLLM_LOG_INFO("KVCM: allocatePools");
    KVCacheManager::allocatePools(dtype, useUvm);
}

void KVCacheWrapper::startScheduling() {
    TLLM_LOG_INFO("KVCM: startScheduling");
    KVCacheManager::startScheduling();
}

/// @brief  Function that computes the number of KV cache blocks needed to advance a request by one or two
/// iterations
/// @param req The request for which we need to calculate the number of needed KV cache blocks
/// @return  The number of blocks
[[nodiscard]] SizeType32 KVCacheWrapper::getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const {
    SizeType32 ret = KVCacheManager::getNeededBlocksOneStep(req, twoStepsLookAhead);
    TLLM_LOG_INFO("KVCM[REQ:%lld]: getNeededBlocksOneStep %d -> %d", (long long)(req.mRequestId), twoStepsLookAhead ? 2 : 0, ret);
    return ret;
}

/// @brief  Function that computes the number of KV cache blocks needed to advance a request to completion (i.e. for
/// maxNewTokens)
/// @param req The request for which we need to calculate the number of needed KV cache blocks
/// @return  The number of blocks
[[nodiscard]] SizeType32 KVCacheWrapper::getNeededBlocksToCompletion(LlmRequest const& req) const {
    SizeType32 ret = KVCacheManager::getNeededBlocksToCompletion(req);
    TLLM_LOG_INFO("KVCM[REQ:%lld]: getNeededBlocksToCompletion -> %d", (long long)(req.mRequestId), ret);
    return ret;
}

void KVCacheWrapper::addContextTokens(SizeType32 seqSlotIdx, SizeType32 numTokens) {
    TLLM_LOG_INFO("KVCM: %d addContextTokens %d", seqSlotIdx, numTokens);
    KVCacheManager::addContextTokens(seqSlotIdx, numTokens);
}

/// @brief Increase size for request at seqSlotIdx. Allocate new KV cache block(s) if needed.
void KVCacheWrapper::addToken(SizeType32 seqSlotIdx) {
    TLLM_LOG_INFO("KVCM: %d addToken", seqSlotIdx);
    KVCacheManager::addToken(seqSlotIdx);
}

/// @brief Add new request to the KV cache manager.
/// @param inputLength Input length for which KV cache need to be allocated.
/// @param beamWidth Beam width for which KV cache need to be allocated.
/// @param llmRequest Optional request to use for KV cache lookup.
/// @details If llmRequest is supplied and KV cache reuse is enabled, try to recover KV cache blocks for
/// inputLength - 1 tokens and populate prepopulatedPromptLen.
void KVCacheWrapper::addSequence(SizeType32 seqSlotIdx, SizeType32 inputLength, SizeType32 beamWidth,
    std::shared_ptr<LlmRequest> const& llmRequest) {
    TLLM_LOG_INFO("KVCM[REQ:%lld]: %d addSequence inputLength=%d beamWidth=%d", (long long)(llmRequest->mRequestId), seqSlotIdx, inputLength, beamWidth);
    KVCacheManager::addSequence(seqSlotIdx, inputLength, beamWidth, llmRequest);
}

void KVCacheWrapper::removeSequence(SizeType32 seqSlotIdx, std::shared_ptr<LlmRequest> const& llmRequest) {
    TLLM_LOG_INFO("KVCM[REQ:%lld]: %d removeSequence", (long long)(llmRequest->mRequestId), seqSlotIdx);
    KVCacheManager::removeSequence(seqSlotIdx, llmRequest);
}

void KVCacheWrapper::schedulingRemoveSequence(SizeType32 seqSlotIdx) {
    TLLM_LOG_INFO("KVCM: %d schedulingRemoveSequence", seqSlotIdx);
    KVCacheManager::schedulingRemoveSequence(seqSlotIdx);
}

[[nodiscard]] runtime::ITensor::UniquePtr KVCacheWrapper::getBlockPoolPointers() const {
    return KVCacheManager::getBlockPoolPointers();
}

void KVCacheWrapper::getBlockOffsetsOfBatch(
    runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const {
    KVCacheManager::getBlockOffsetsOfBatch(output, firstBatchSlotIdx, batchSize, beamWidth);
    TLLM_LOG_INFO("KVCM: %d getBlockOffsetsOfBatch, batchSize=%d, beamWidth=%d", firstBatchSlotIdx, batchSize, beamWidth);
}

SizeType32 KVCacheWrapper::copyBlockOffsets(
    runtime::ITensor& output, SizeType32 outputSlotOffset, SizeType32 seqSlotIdx, SizeType32 beamWidth) const {
    SizeType32 ret = KVCacheManager::copyBlockOffsets(output, outputSlotOffset, seqSlotIdx, beamWidth);
    TLLM_LOG_INFO("KVCM: %d getBlockOffsetsOfBatch, outputSlotOffset=%d, beamWidth=%d -> %d", seqSlotIdx, outputSlotOffset, beamWidth, ret);
    return ret;
}

[[nodiscard]] std::tuple<SizeType32, SizeType32> const KVCacheWrapper::calculateMaxNumBlocks(KvCacheConfig const& config,
    nvinfer1::DataType dtype, tensorrt_llm::runtime::ModelConfig const& modelConfig,
    tensorrt_llm::runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager) {
    return KVCacheManager::calculateMaxNumBlocks(config, dtype, modelConfig, worldConfig, bufferManager);
}

void KVCacheWrapper::removeToken(SizeType32 seqSlotIdx) {
    TLLM_LOG_INFO("KVCM: %d removeToken", seqSlotIdx);
    KVCacheManager::removeToken(seqSlotIdx);
}
void KVCacheWrapper::rewindKVCache(SizeType32 seqSlotIdx, SizeType32 rewindLengths) {
    TLLM_LOG_INFO("KVCM: %d rewindKVCache %d", seqSlotIdx, rewindLengths);
    KVCacheManager::rewindKVCache(seqSlotIdx, rewindLengths);
}

[[nodiscard]] GenerationRequest const& KVCacheWrapper::getSequence(SizeType32 seqSlotIdx) const {
    TLLM_LOG_INFO("KVCM: %d getSequence", seqSlotIdx);
    return KVCacheManager::getSequence(seqSlotIdx);
}

[[nodiscard]] SizeType32 KVCacheWrapper::getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock) {
    SizeType32 ret = KVCacheManager::getSinkBubbleLength(sinkTokenLen, tokensPerBlock);
    TLLM_LOG_INFO("KVCM: getSinkBubbleLength %d %d -> %d", sinkTokenLen, tokensPerBlock, ret);
    return ret;
}

[[nodiscard]] SizeType32 KVCacheWrapper::getMaxAttentionWindowUpperBound(SizeType32 blocksInPrimaryPool,
    SizeType32 tokensPerBlock, SizeType32 maxBeamWidth, SizeType32 sinkTokenLen, bool useOneMoreBlock) {
    SizeType32 ret = KVCacheManager::getMaxAttentionWindowUpperBound(blocksInPrimaryPool, tokensPerBlock, maxBeamWidth, sinkTokenLen, useOneMoreBlock);
    TLLM_LOG_INFO("KVCM: getMaxAttentionWindowUpperBound %d %d %d %d %d -> %d", blocksInPrimaryPool, tokensPerBlock, maxBeamWidth, sinkTokenLen, useOneMoreBlock ? 1 : 0, ret);
    return ret;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager

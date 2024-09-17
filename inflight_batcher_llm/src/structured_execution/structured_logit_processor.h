/* structured_logit_processor.h
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

#pragma once

#include <vector>
#include <optional>
#include <string>
#include <unordered_map>

#include <unistd.h>
#include <fcntl.h>

#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include "structured_execution_engine.h"
#include "structured_execution_safetensors.h"

struct CUstream_st;

namespace triton::backend::inflight_batcher_llm {

class StructuredLogitProcessorRequestState {
protected:
    unsigned num_input_tokens = 0;
    unsigned num_processed_tokens = 0;

    explicit StructuredLogitProcessorRequestState(unsigned num_input_tokens, int end_id)
        : num_input_tokens(num_input_tokens), num_processed_tokens(0), end_id(end_id) {
    }

protected:
    virtual void executeToken(unsigned token, bool is_prefill) = 0;

public:
    int end_id = 0;

    virtual int getNextTensorIndex() = 0;

    void executeTokens(const tensorrt_llm::executor::BeamTokens &ids_vec) {
        // TODO: implement beam_width > 1
        const tensorrt_llm::executor::VecTokens &ids = ids_vec[0];
        for (; num_processed_tokens < (unsigned)ids.size(); num_processed_tokens++) {
            executeToken(ids[num_processed_tokens], num_processed_tokens < num_input_tokens);
        }
    }

    virtual ~StructuredLogitProcessorRequestState() = default;
};

static_assert(sizeof(bool) == 1);
using tensorrt_llm::runtime::SizeType32;

// Defined in structured_logit_processor.cu
template <typename T>
void invokeApplyLogitMask(
    T* const* logits, SizeType32 vocabSize, std::int32_t *states, std::int32_t *end_ids, SizeType32 batchSize,
    const bool *maskVector, SizeType32 maskTokenSizePadded, struct CUstream_st* stream);

class StructuredBatchedLogitProcessor;

class FreeStateHolder
{
    StructuredBatchedLogitProcessor *owner;
    uint32_t file_id;
    uint32_t client_epoch;

public:
    FreeStateHolder(StructuredBatchedLogitProcessor *owner, uint32_t file_id, uint32_t client_epoch)
        : owner(owner), file_id(file_id), client_epoch(client_epoch) {
    }
    ~FreeStateHolder();
    uint32_t get_file_id() {
        return file_id;
    }
    tensorrt_llm::executor::IdType get_client_id() {
        return (tensorrt_llm::executor::IdType)((((uint64_t)client_epoch) << 32) | file_id);
    }
};

class StructuredBatchedLogitProcessor
{
protected:
    using IdType = tensorrt_llm::executor::IdType;
    using Tensor = tensorrt_llm::executor::Tensor;
    using Shape = tensorrt_llm::executor::Shape;
    using DataType = tensorrt_llm::executor::DataType;

    virtual const uint8_t *getStateMaskData() = 0;
    virtual size_t getNumStateKeys() = 0;
    virtual size_t getPaddedTokenSize() = 0;

private:

    std::vector<std::unique_ptr<StructuredLogitProcessorRequestState>> state_objects;
    std::vector<int> state_fds;
    std::vector<uint32_t> client_epochs;
    std::vector<uint32_t> next_client_epochs;
    std::vector<uint32_t> free_state_list;

    Tensor current_states_cpu;
    Tensor logit_ptrs_cpu;
    Tensor current_states_gpu;
    Tensor end_ids_cpu;
    Tensor logit_ptrs_gpu;
    Tensor state_mask_tensor_gpu;
    Tensor end_ids_gpu;

    StructuredLogitProcessorRequestState* getRequestState(uint64_t client_id) {
        uint32_t this_epoch_id = (uint32_t)(client_id >> 32);
        uint32_t file_id = (uint32_t)client_id;
        TLLM_CHECK(file_id >= 0 && file_id < client_epochs.size());
        if (client_epochs[file_id] == this_epoch_id) {
            return state_objects[file_id].get();
        }
        client_epochs[file_id] = this_epoch_id;
        state_objects[file_id].reset();
        int fd = state_fds[file_id];
        int num_input_tokens = -1;
        int end_id = -1;
        int str_len = -1;
        read(fd, &num_input_tokens, sizeof(int));
        read(fd, &end_id, sizeof(int));
        read(fd, &str_len, sizeof(int));
        if (str_len <= 0 || num_input_tokens < 0 || end_id < 0) {
            return nullptr;
        }
        char *json_buf = new char[str_len];
        bool success = (int)read(fd, json_buf, str_len) == str_len;
        lseek(fd, 0, SEEK_SET);
        if (success) {
            std::string structuredExecutionData (json_buf, json_buf + str_len);
            nlohmann::json json_settings = nlohmann::json::parse(structuredExecutionData);
            state_objects[file_id] = createRequestState(num_input_tokens, end_id, json_settings);
        }
        delete []json_buf;

        return state_objects[file_id].get();
    }

    template<class T>
    void executeTyped(
        std::vector<IdType> const&req_ids_batch,
        std::vector<Tensor> &logits_batch,
        std::vector<std::reference_wrapper<tensorrt_llm::executor::BeamTokens const>> const&ids_batch,
        tensorrt_llm::executor::StreamPtr const&stream_ptr,
        std::vector<std::optional<IdType>> const&client_ids_batch)
    {
        size_t state_key_count = getNumStateKeys();
        size_t padded_vocab_size = getPaddedTokenSize();
        if (!current_states_gpu)
        {
            current_states_gpu = current_states_cpu.copyToGpu(stream_ptr);
            end_ids_gpu = end_ids_cpu.copyToGpu(stream_ptr);
            logit_ptrs_gpu = logit_ptrs_cpu.copyToGpu(stream_ptr);
            state_mask_tensor_gpu = Tensor::of(DataType::kBOOL, const_cast<uint8_t*>(getStateMaskData()),
                Shape({static_cast<Shape::DimType64>(padded_vocab_size * state_key_count)})).copyToGpu(stream_ptr);
        }
        bool hasJson = false;
        int vocab_size = (int) logits_batch[0].getSize();
        {
            int* current_states_cpu_data = reinterpret_cast<int*>(current_states_cpu.getData());
            int* end_ids_cpu_data = reinterpret_cast<int*>(end_ids_cpu.getData());
            T** logit_ptrs_cpu_data = reinterpret_cast<T**>(logit_ptrs_cpu.getData());
            for (std::size_t i = 0, sz = req_ids_batch.size(); i < sz; ++i)
            {
                current_states_cpu_data[i] = -1;
                logit_ptrs_cpu_data[i] = reinterpret_cast<T*>(logits_batch[i].getData());
                uint64_t client_id = (uint64_t)(client_ids_batch[i].value_or((uint64_t)-1));
                if (client_id != (uint64_t)-1)
                {
                    StructuredLogitProcessorRequestState* state;
                    state = getRequestState(client_id);
                    hasJson = true;
                    state->executeTokens(ids_batch[i]);
                    end_ids_cpu_data[i] = state->end_id;
                    int this_state = state->getNextTensorIndex();
                    if (this_state >= 0 && this_state < (int)state_key_count)
                    {
                        current_states_cpu_data[i] = this_state;
                    }
                }
            }
        }
        if (hasJson)
        {
            logit_ptrs_gpu.setFrom(logit_ptrs_cpu, stream_ptr);
            current_states_gpu.setFrom(current_states_cpu, stream_ptr);
            end_ids_gpu.setFrom(end_ids_cpu, stream_ptr);
            invokeApplyLogitMask<T>(
                reinterpret_cast<T**>(logit_ptrs_gpu.getData()), vocab_size,
                reinterpret_cast<int*>(current_states_gpu.getData()),
                reinterpret_cast<int*>(end_ids_gpu.getData()), req_ids_batch.size(),
                reinterpret_cast<const bool*>(state_mask_tensor_gpu.getData()), padded_vocab_size,
                reinterpret_cast<struct CUstream_st*>(stream_ptr->get()));
        }
    }
protected:
    virtual std::unique_ptr<StructuredLogitProcessorRequestState> createRequestState(
        int num_input_tokens, int end_id, const json &json_settings) = 0;

public:
    explicit StructuredBatchedLogitProcessor(int max_batch_size)
        : current_states_cpu(Tensor::cpu(DataType::kINT32, Shape{max_batch_size})),
          logit_ptrs_cpu(Tensor::cpu(DataType::kINT64, Shape{max_batch_size})),
          end_ids_cpu(Tensor::cpu(DataType::kINT32, Shape{max_batch_size}))
    {
        state_objects.resize(max_batch_size);
        client_epochs.resize(max_batch_size, (uint32_t)-1);
        next_client_epochs.resize(max_batch_size);
        for (uint32_t file_id = 0; file_id < (uint32_t)max_batch_size; file_id++) {
            char filename[100] = {0};
            snprintf(filename, 100, "/dev/shm/json_%d_data", file_id);
            int fd = open(filename, O_RDWR | O_CREAT, 0666);
            if (fd == -1) {
                TLLM_LOG_ERROR("Unable to open output file %s for w+", filename);
            }
            state_fds.push_back(fd);
            free_state_list.push_back(file_id);
        }

        //cudaMalloc(&current_states_device, sizeof(int) * max_batch_size);
        current_states_cpu.setZero();
        logit_ptrs_cpu.setZero();
        end_ids_cpu.setZero();
    }
    virtual ~StructuredBatchedLogitProcessor() {
        cudaFree(current_states_gpu.getData());
        cudaFree(logit_ptrs_gpu.getData());
        cudaFree(end_ids_gpu.getData());
        for (int fd : state_fds) {
            close(fd);
        }
    }

    virtual bool isInitialized() = 0;

    std::unique_ptr<FreeStateHolder> startRequest(int num_input_tokens, int end_id, const json &json_settings) {
        if (free_state_list.empty()) {
            TLLM_LOG_ERROR("Out of states: Unable to create a json logit processor.");
            return nullptr;
        }
        // Create throwaway object to ensure it is a valid request.
        if (createRequestState(num_input_tokens, end_id, json_settings)) {
            std::string json_str = to_string(json_settings);
            if (json_str.size() >= INT_MAX || json_str.empty()) {
                return nullptr;
            }
            uint32_t file_id = free_state_list.back();
            free_state_list.pop_back();
            uint32_t client_epoch = ++next_client_epochs[file_id];
            std::unique_ptr<FreeStateHolder> ret = std::make_unique<FreeStateHolder>(this, file_id, client_epoch);
            int str_len = (int)json_str.size();
            int fd = state_fds[ret->get_file_id()];
            write(fd, &num_input_tokens, sizeof(int));
            write(fd, &end_id, sizeof(int));
            write(fd, &str_len, sizeof(int));
            write(fd, json_str.data(), str_len);
            lseek(fd, 0, SEEK_SET);
            return ret;
        }
        return nullptr;
    }

    void release_file_id(uint32_t file_id) {
        free_state_list.push_back(file_id);
    }

    void operator()(
        std::vector<IdType> const&req_ids_batch,
        std::vector<Tensor> &logits_batch,
        std::vector<std::reference_wrapper<tensorrt_llm::executor::BeamTokens const>> const&ids_batch,
        tensorrt_llm::executor::StreamPtr const&stream_ptr,
        std::vector<std::optional<IdType>> const&client_ids_batch)
    {
        switch (logits_batch[0].getDataType()) {
        case DataType::kBF16:
            executeTyped<__nv_bfloat16>(req_ids_batch, logits_batch, ids_batch, stream_ptr, client_ids_batch);
            break;
        case DataType::kFP16:
            executeTyped<half>(req_ids_batch, logits_batch, ids_batch, stream_ptr, client_ids_batch);
            break;
        case DataType::kFP32:
            executeTyped<float>(req_ids_batch, logits_batch, ids_batch, stream_ptr, client_ids_batch);
            break;
        default:
            break;
        }
    }

};

inline FreeStateHolder::~FreeStateHolder() {
    owner->release_file_id(file_id);
}

}

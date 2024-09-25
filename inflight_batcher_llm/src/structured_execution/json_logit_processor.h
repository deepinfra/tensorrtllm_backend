/* json_logit_processor.h
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

#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/executor/executor.h"

#include "structured_execution_engine.h"
#include "structured_execution_safetensors.h"
#include "structured_logit_processor.h"

namespace structured_execution
{
using namespace triton::backend::inflight_batcher_llm;
using nlohmann::json;

class JsonBatchedLogitProcessor;

class JsonRequestState : public StructuredLogitProcessorRequestState
{
    std::unique_ptr<StructureExecutionEngine> engine;

    bool prompt_completion = false;
    bool embedded = false;
    int lock_in_state = 1;
    std::string type_str;

public:
    enum
    {
        DEFAULT_INDENT_SIZE = 4
    };

    JsonRequestState(int num_input_tokens, int end_id,
        const std::shared_ptr<PrecalculatedStructureGraph> &precalculated_structure_graph,
        const json &json_settings);

    void executeToken(unsigned token, bool is_prefill) override;
    int getNextTensorIndex() override;
};

class JsonBatchedLogitProcessor : public StructuredBatchedLogitProcessor
{
    std::thread initialization_thread;
    std::shared_ptr<PrecalculatedStructureGraph> precalculated_structure_graph;

public:
    JsonBatchedLogitProcessor(int max_batch_size, const std::string &safetensors_filename)
    : StructuredBatchedLogitProcessor(max_batch_size)
    {
        FILE* fp = fopen(safetensors_filename.c_str(), "rb");

        initialization_thread = std::thread([this, safetensors_filename, fp]() {
            precalculated_structure_graph = parse_graph(fp);
            fclose(fp);
            if (precalculated_structure_graph) {
                TLLM_LOG_INFO("Successfully loaded logit processor at %s", safetensors_filename.c_str());
            } else {
                TLLM_LOG_ERROR("Failed to read logit processor input tensors %s: %d", safetensors_filename.c_str(), errno);
            }
        });
    }

    virtual ~JsonBatchedLogitProcessor() {
        initialization_thread.join(); // Prevents destructor crash due to initialization failure.
    }

    const uint8_t *getStateMaskData() override {
        return precalculated_structure_graph->precalculated_vectors_tensor.get();
    }
    size_t getNumStateKeys() override {
        return precalculated_structure_graph->precalculated_state_keys.size();
    }
    size_t getPaddedTokenSize() override {
        return precalculated_structure_graph->tokenizer_data->padded_token_size;
    }
    bool isInitialized() override {
        if (initialization_thread.joinable()) {
            initialization_thread.join();
        }
        return (bool)precalculated_structure_graph;
    }

    std::unique_ptr<StructuredLogitProcessorRequestState> createRequestState(
        int num_input_tokens, int end_id, const json &json_settings) override
    {
        if (initialization_thread.joinable()) {
            initialization_thread.join();
        }
        if (!isInitialized()) {
            return {};
        }
        if (!json_settings.contains("type")) {
            return {};
        }
        if (!json_settings["type"].is_string() || (
                json_settings["type"].get<std::string>().find("json") != 0 &&
                json_settings["type"].get<std::string>().find("markdown_json") != 0 &&
                json_settings["type"].get<std::string>() != "function_call")) {
            return {};
        }
        return std::make_unique<JsonRequestState>(
            num_input_tokens, end_id, precalculated_structure_graph, json_settings);
    }
};

inline JsonRequestState::JsonRequestState(int num_input_tokens, int end_id,
        const std::shared_ptr<PrecalculatedStructureGraph> &precalculated_structure_graph,
        const json &json_settings)
    : StructuredLogitProcessorRequestState(num_input_tokens, end_id)
{
    int indent_space_size = DEFAULT_INDENT_SIZE;
    if (json_settings.contains("indent") && json_settings["indent"].is_number_unsigned()) {
        indent_space_size = json_settings["indent"].get<int>();
        if (indent_space_size < 0 || indent_space_size > 8) {
            indent_space_size = DEFAULT_INDENT_SIZE;
        }
    }
    engine = std::make_unique<StructureExecutionEngine>(precalculated_structure_graph, indent_space_size);
    type_str = json_settings["type"].get<std::string>();
    if (embedded && type_str == "function_call") {
        lock_in_state = 10;
    }
    if (json_settings.contains("embedded") && json_settings["embedded"].is_boolean()) {
        embedded = json_settings["embedded"].get<bool>();
    }
    if (json_settings.contains("prompt_completion") && json_settings["prompt_completion"].is_boolean()) {
        prompt_completion = json_settings["prompt_completion"].get<bool>();
    }
    engine->set_state(type_str);
}

inline void JsonRequestState::executeToken(unsigned token, bool is_prefill) {
    if (!is_prefill || prompt_completion) {
        SingleNodeKey new_key = engine->execute_tok(token);
        if (new_key.structName == InternedString()) {
            if (embedded && lock_in_state != 1 && engine->struct_stack.size() == 1 && engine->struct_stack.back().pos <= lock_in_state) {
                engine->set_state(type_str);
            }
        }
    }
    if ((embedded || (is_prefill && prompt_completion)) && engine->reached_end()) {
        engine->set_state(type_str);
    }
}

inline int JsonRequestState::getNextTensorIndex() {
    int acceptable_token_index = (int)engine->get_logit_weights_index();
    if (embedded && engine->struct_stack.size() == 1 && engine->struct_stack.back().pos <= lock_in_state) {
        // Allow any token until we are committed.
        acceptable_token_index = -1;
    }
    return acceptable_token_index;
}


}

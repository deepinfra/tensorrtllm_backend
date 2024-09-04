/* structure_execution_safetensors.h
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

#include <locale>
#include <string>

#include "structured_execution_engine.h"
//#include <nlohmann/json.hpp>
#include <nlohmann/json.hpp>

using namespace nlohmann;

namespace std {
    struct char_to_char32 : public std::codecvt<char32_t, char, std::mbstate_t> {
        using std::codecvt<char32_t, char, std::mbstate_t>::codecvt;
        ~char_to_char32() = default;
    };

    inline void to_json(json &j, const std::u32string &str_vec) {
        std::wstring_convert<char_to_char32, char32_t> utf32_to_utf8_converter;
        j = json(utf32_to_utf8_converter.to_bytes(str_vec));
    }

    inline void from_json(const json &j, std::u32string &str_vec) {
        if (!j.is_string()) {
            return;
        }
        std::wstring_convert<char_to_char32, char32_t> utf32_to_utf8_converter;
        str_vec = utf32_to_utf8_converter.from_bytes(j.template get<std::string>());
    }
}

namespace structured_execution {
    inline void to_json(json &j, const SingleNodeKey &nodeKey) {
        if (nodeKey.nodeType != InternedString::EMPTY) {
            j = json {nodeKey.structName.stringId, nodeKey.pos, nodeKey.nodeType.stringId};
        } else {
            j = json {nodeKey.structName.stringId, nodeKey.pos};
        }
    }

    inline void to_json(json &j, const StructOperation &op) {
        switch (op.type) {
            case StructOperation::PUSH:
                to_json(j, op.pushStruct);
                break;
            case StructOperation::APPEND_CHAR:
                j = (int64_t)op.appendChar;
                break;
            case StructOperation::POP:
            default:
                j = nullptr;
        }
    }

    /*
    inline void to_json(json &j, const PrecalculatedStructureGraph::PerTokenDataPair &graph_value) {
        j = json::array({
            graph_value.first.first,
            graph_value.first.second,
            graph_value.second.first,
            graph_value.second.second
        });
    }
    */

    inline void from_json(const json &j, SingleNodeKey &nodeKey) {
        if (j.is_array()) {
            if (j.size() >= 2) {
                j[0].get_to(nodeKey.structName.stringId);
                j[1].get_to(nodeKey.pos);
                nodeKey.nodeType = InternedString::EMPTY;
                if (j.size() >= 3) {
                    j[2].get_to(nodeKey.nodeType.stringId);
                }
            }
        }
    }

    inline void from_json(const json &j, StructOperation &op) {
        if (j.is_null()) {
            op = StructOperation::pop();
        } else if (j.is_number()) {
            op.type = StructOperation::APPEND_CHAR;
            int64_t char_type;
            j.get_to(char_type);
            op = StructOperation::append((char32_t)char_type);
        } else {
            op.type = StructOperation::PUSH;
            SingleNodeKey push_key;
            j.get_to(push_key);
            op = StructOperation::push(push_key);
        }
    }

    /*
    inline void from_json(const json &j, PrecalculatedStructureGraph::PerTokenDataPair &graph_value) {
        if (j.is_array() && j.size() == 4) {
            NodeKeyTuple key_f;
            unsigned key_s;
            OpTuple value_f;
            SingleNodeKey value_s;
            j[0].get_to(key_f);
            j[1].get_to(key_s);
            j[2].get_to(value_f[0]);
            j[3].get_to(value_s);
            graph_value = std::make_pair(std::make_pair(std::move(key_f), key_s), std::make_pair(std::move(value_f), value_s));
            PrecalculatedStructureGraph::PerTokenDataPair new_pair = std::make_pair(std::make_pair(std::move(key_f), key_s), std::make_pair(std::move(value_f), value_s));
            //graph_value = std::move(new_pair);
            graph_value.first.first = new_pair.first.first;
            graph_value.first.second = new_pair.first.second;
            graph_value.second.first = new_pair.second.first;
            graph_value.second.second = new_pair.second.second;
        };
    }
    */

    inline void to_json(json &j, const PrecalculatedStructureGraph &graph) {
        std::vector<PrecalculatedStructureGraph::PerTokenDataPair> graph_sorted(graph.token_output_graph.begin(),
                                                                                     graph.token_output_graph.end());
        std::sort(graph_sorted.begin(), graph_sorted.end());
        j = {
                {"shape", {graph.precalculated_state_keys.size(), graph.tokenizer_data->padded_token_size}},
                {"dtype", "BOOL"},
                {"data_offsets", {0, graph.precalculated_state_keys.size() * graph.tokenizer_data->padded_token_size}},
                {"token_strings", graph.token_strings},
                {"intern_table", graph.intern_table.table},
                {"root_node_key", graph.root_node_key},
                {"history_length", graph.history_length},
                {"precalculated_state_keys", graph.precalculated_state_keys},
                {"token_output_graph", graph_sorted}
        };
    }
    inline bool write_graph (FILE *fp, const PrecalculatedStructureGraph &graph) {
        json j = {{"precalculated_structure_graph", graph}};
        std::string json_str = j.dump();
        uint64_t json_len = json_str.size();
        if (fwrite(&json_len, 1, 8, fp) != 8) {
            return false;
        }
        if (fwrite(json_str.data(), 1, json_len, fp) != json_len) {
            return false;
        }
        uint64_t data_len = graph.precalculated_state_keys.size() * graph.tokenizer_data->padded_token_size;
        //fwrite(&data_len, 1, 8, fp);
        if (fwrite(graph.precalculated_vectors_tensor.get(), 1, data_len, fp) != data_len) {
            return false;
        }
        return true;
    }
    inline std::shared_ptr<PrecalculatedStructureGraph> parse_graph (FILE *fp) {
        std::unique_ptr<PrecalculatedStructureGraph> ret;
        uint64_t json_length = 0;
        fseek(fp, 0, SEEK_SET);
        if (fread(&json_length, 1, 8, fp) != 8) {
            return ret;
        }
        if (json_length > 100000000 || json_length == 0) {
            return ret;
        }
        fseek(fp, 0, SEEK_END);
        size_t data_length = ftell(fp) - json_length - 8;
        fseek(fp, 8, SEEK_SET);

        std::string json_data(json_length, ' ');
        if (fread(&(json_data[0]), 1, json_length, fp) != json_length) {
            return ret;
        }
        // FIXME: I'm leaking `j_wrapped` because the json destructor is a bit slow.
        // (and I don't want to expose the json implementation to other parts of the system)
        json *j_wrapped = new json(json::parse(json_data));
        const json &j = j_wrapped->at("precalculated_structure_graph");

        std::vector<std::u32string> token_strings;
        j.at("token_strings").get_to(token_strings);
        std::unique_ptr<TokenizerData> tokenizer_data = std::make_unique<TokenizerData>(token_strings);

        std::unique_ptr<InternTable> intern_table = std::make_unique<InternTable>();
        std::vector<std::u32string> intern_table_strings;
        j.at("intern_table").get_to(intern_table_strings);
        for (const std::u32string &str: intern_table_strings) {
            if (str.empty()) {
                continue;
            }
            intern_table->intern(str);
        }
        SingleNodeKey root_node_key;
        j.at("root_node_key").get_to(root_node_key);
        std::vector<NodeKeyTuple> precalculated_state_keys;
        j.at("precalculated_state_keys").get_to(precalculated_state_keys);
        std::vector<uint64_t> shape;
        unsigned history_length = 0;
        j.at("history_length").get_to(history_length);

        std::vector<PrecalculatedStructureGraph::PerTokenDataPair> token_output_graph_keys;
        j.at("token_output_graph").get_to(token_output_graph_keys);

        j.at("shape").get_to(shape);
        if (shape.size() != 2 || shape[0] != precalculated_state_keys.size() ||
            shape[1] != tokenizer_data->padded_token_size) {
            return ret;
        }
        std::vector<uint64_t> data_offsets;
        j.at("data_offsets").get_to(data_offsets);
        if (data_offsets.size() != 2 || data_offsets[0] != 0 ||
            data_offsets[1] != shape[0] * shape[1]) {
            return ret;
        }
        std::string dtype;
        j.at("dtype").get_to(dtype);
        if (dtype != "BOOL") {
            return ret;
        }

//        uint64_t data_length = 0;
//        fread(&data_length, 1, 8, fp);
        if (data_length < data_offsets[1]) {
            return ret;
        }
        std::unique_ptr<uint8_t[]> precalculated_vectors_tensor (new uint8_t[data_length]);
        if (fread(precalculated_vectors_tensor.get(), 1, data_length, fp) != data_length) {
            return ret;
        }

        return std::make_shared<PrecalculatedStructureGraph>(
                std::move(tokenizer_data), std::move(intern_table),
                root_node_key, history_length,
                std::move(precalculated_state_keys), std::move(precalculated_vectors_tensor),
                std::move(token_output_graph_keys));
    }
}

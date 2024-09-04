/* structure_execution_engine.cpp
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

#define DEBUG_INTERNED_STRINGS

#include <algorithm>
#include <cstring>
#include <cctype>

#include "structured_execution_engine.h"

namespace structured_execution {

using namespace StructureDebug;

int VERBOSE = 0;

class ZeroTerminatedHasher {
public:
    std::size_t operator()(const char32_t *cdata) const {
        uint64_t acc = 14695981039346656037ULL;
        while (*cdata) {
            std::size_t next = *cdata;
            acc = (acc ^ next) * 1099511628211ULL;
            cdata++;
        }
        return acc;
    }
};

ParserNode *ParserNode::n(char32_t chr, ParserStructureStack *stack) {
    int match = which_match(chr);
    ParserNode *ret = next(chr, stack, match);
    //char32_t d[2] = {chr, 0};
    if (ret != nullptr) {
        //std::cout << "-> " << *this << " n(" << d << ", stack, " << match << ")" << std::endl << " -> " << *ret << std::endl;
    } else {
        //std::cout << "-> n(" << d << ", stack, " << match << ")" << std::endl << " -> nullptr" << std::endl;
    }
    return ret;
}

void Structure::_construct_indices_recursive(ParserNode *node, ParserNode *optional_root, const std::u32string &path) {
    int idx = (int)this->indices.size();
    if (node->as_structure_node()) {
        is_string = false;
    }
    // Instantiate class name so string indices are compatible between VERBOSE on and off.
    (void)node->get_class_name();
    if (VERBOSE >= 1) {
        std::cout << name << "._construct_indicices_recursive: " << *node;
        if (optional_root) {
            std::cout << " from optional_root " << *optional_root;
        }
    }
    if (optional_root && !node->as_structure_node()) {
        if (VERBOSE >= 1) {
            std::cout << " (not reachable due to optional_root)";
        }
        this->indices.emplace_back(*node);
        node->index = idx;
        node->reachable = false;
    } else {
        if (VERBOSE >= 1) {
            std::cout << " (reachable)";
        }
        this->indices.emplace_back(*node);
        node->index = idx;
        node->reachable = true;
    }

    if (node->is_optional_root()) {
        if (VERBOSE >= 1) {
            std::cout << " (update optional_root)";
        }
        optional_root = node;
    }
    if (VERBOSE >= 1) {
        std::cout << std::endl;
    }

    unsigned int child_count = node->get_child_count();
    if (child_count > 0) {
        for (int i = 0; i < (int)child_count; i++) {
            node->get_child(i)->parent = node;
            node->get_child(i)->path_key_idx = i;
            node->get_child(i)->path = path + U"/" + to_u32string(i) + U"_" + node->get_child(i)->get_debug_name();
            if (i != 0 && node->as_sequence_node()) {
                optional_root = nullptr;
            }
            this->_construct_indices_recursive(node->get_child(i), optional_root, node->get_child(i)->path);
        }
    } else {
        ParserNode * only_child = node->get_child(0);
        if (only_child) {
            only_child->path_key = node->get_class_name();
            only_child->parent = node;
            only_child->path = path + U"/" + only_child->get_debug_name();
            this->_construct_indices_recursive(only_child, optional_root, only_child->path);
        }
    }
}

void Structure::construct_indices() {
    end_structure->path_key = end_string;
    indices.emplace_back(*end_structure);
    root_node->reachable = true;
    root_node->path_key = root_string;
    (void)get_class_name(); // Instantiate class name so string indices are compatible between VERBOSE on and off.
    root_node->path = name.get(*intern_table) + U":" + root_node->get_debug_name() + U"/";
    _construct_indices_recursive(root_node.get(), nullptr,
                                 name.get(*intern_table) + U":" + root_node->get_debug_name());
    indices.emplace_back(*this);
    //index = indices.size() - 1;
}

/*
class Structure:
    def get_path(self, node: ParserNode) -> tuple:
        return this->indices[this->node_to_index[node]][1]

    def path_to_node(self, path: PathType) -> ParserNode:
        return this->indices[this->path_to_index[path]][0]

    def index_to_node(self, idx: int) -> ParserNode:
        return this->indices[idx][0]

    def index_to_path(self, idx: int) -> PathType:
        return this->indices[idx][1]

    // Not used in production, just for testing
    def match(self, inp: str) -> Any:  # tuple[Any, Optional[ParserNode]]:
        structure_stack: ParserStructureStack = ParserStructureStack(self)
        exp: Optional[ParserNode] = this->root_node
        for chr in inp:
            if exp is nullptr:
                return nullptr
            exp = this->tick(structure_stack, exp, chr)
        if exp is nullptr:
            return nullptr
        this->tick(structure_stack, exp, '')
        return structure_stack.construct_ast()

    def __repr__(self) -> str:
        return 'StructureClass(%r)' % (this->name,)

    def __str__(self) -> str:
        return 'StructureClass(%r, %r)' % (this->name, this->root_node)
    */

typedef std::pair<SingleNodeKey, const char32_t *> TokensCalculatedKey;


struct CalculatedToken {
    OpTuple ops;
    StructureNode &node;
    ParserNode &parser_node;
    const char32_t *chars;

    CalculatedToken(OpTuple ops, StructureNode &node, ParserNode &parser_node, const char32_t *chars)
            : ops(std::move(ops)), node(node), parser_node(parser_node), chars(chars) {
    }

    bool operator==(const CalculatedToken &oth) const {
        if (&parser_node != &oth.parser_node || &node != &oth.node || ops != oth.ops) {
            return false;
        }
        for (int i = 0; chars[i] != 0 || oth.chars[i] != 0; i++) {
            if (chars[i] != oth.chars[i]) {
                return false;
            }
        }
        return true;
    }
};

struct TokensCalculatedKeyHasher {
    std::size_t operator()(const TokensCalculatedKey &arg) const {
        return CustomHasher()(arg.first) * 7 + ZeroTerminatedHasher()(arg.second) * 11;
    }

    std::size_t operator()(const CalculatedToken &arg) const {
        return CustomHasher()(arg.ops) * 5 + std::hash<void *>{}(&arg.node) * 7 +
               std::hash<void *>{}(&arg.parser_node) * 11 + ZeroTerminatedHasher()(arg.chars) * 11;
    }

};

struct TokensCalculatedKeyEq {
    bool operator()(const TokensCalculatedKey &a, const TokensCalculatedKey &b) const {
        if (a.first != b.first) {
            return false;
        }
        const char32_t *astr = a.second;
        const char32_t *bstr = b.second;
        while (*astr && *bstr) {
            if (*astr != *bstr) {
                return false;
            }
            ++astr;
            ++bstr;
        }
        if (*astr || *bstr) {
            return false;
        }
        return true;
    }

    bool operator()(const CalculatedToken &a, const CalculatedToken &b) const {
        if (a.ops != b.ops) {
            return false;
        }
        if (&a.node != &b.node) {
            return false;
        }
        if (&a.parser_node != &b.parser_node) {
            return false;
        }
        const char32_t *astr = a.chars;
        const char32_t *bstr = b.chars;
        while (*astr && *bstr) {
            if (*astr != *bstr) {
                return false;
            }
            ++astr;
            ++bstr;
        }
        if (*astr || *bstr) {
            return false;
        }
        return true;
    }

};

std::vector<std::reference_wrapper<ParserNode>> NodeStructureGraph::get_all_basic_entry_points() const {
    std::vector<std::reference_wrapper<ParserNode>> entry_points;
    for (const std::unordered_map<InternedString, std::unique_ptr<Structure>>::value_type &kv: structures) {
        Structure &structure = *kv.second;
        if (VERBOSE >= 1) {
            std::cout << structure.name << std::endl;
        }
        for (size_t idx = 0; idx < structure.indices.size(); idx++) {
            ParserNode & node = structure.indices[idx];
            if (node.reachable) {
                if (VERBOSE >= 1) {
                    std::cout << structure.name << " Adding basic entry point " << idx << " " << node << std::endl;
                }
                entry_points.emplace_back(node);
            } else {
                // Skips about 40% of entry points
                if (VERBOSE >= 2) {
                    std::cout << structure.name << " Skipping " << idx << " " << node << std::endl;
                }
            }
        }
    }
    // if VERBOSE >= 1:
    //     print("Found", len(entry_points), "entry points")
    return entry_points;
}

PrecalculatedRawGraph::PrecalculatedRawGraph(const TokenizerData &tokenizer_data,
                                             const NodeStructureGraph &node_graph) {
    if (VERBOSE > 0) {
        intern_table = &node_graph.intern_table;
    }

    PrecalculatedRawGraph &precalculated_raw_graph = *this;
//    InternedString EndNode = node_graph.intern_string(U"EndNode");

    if (VERBOSE >= 1) {
        std::cout << "Calculating structure graph" << std::endl;
    }
    std::vector<const char32_t *> token_char_lists;
    std::vector<std::pair<int, std::u32string>> stripped_tokens_to_process; //: list[tuple[int, str]] = []
    stripped_tokens_to_process.reserve(tokenizer_data.token_strings.size());
    for (unsigned token_id = 0; token_id < tokenizer_data.token_strings.size(); token_id++) {
        std::u32string s = tokenizer_data.token_strings[token_id];
        if (s.empty()) {
            // special or eos token are empty string.
            continue;
        }
        // NOTE: No longer stripping tokens. Caused trouble with auto indent and newlines.
        //s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](char32_t ch) { return !std::isspace(ch); }));
        //s.erase(std::find_if(s.rbegin(), s.rend(), [](char32_t ch) { return !std::isspace(ch); }).base(), s.end());
        stripped_tokens_to_process.emplace_back(token_id, s);
        token_char_lists.emplace_back(stripped_tokens_to_process.back().second.c_str());
    }

    std::vector<std::reference_wrapper<ParserNode>> basic_entry_points = node_graph.get_all_basic_entry_points();

    std::unordered_map<StructureNode *, std::vector<std::pair<std::reference_wrapper<ParserNode>, std::reference_wrapper<StructureNode>>>> entry_point_parents;
    for (ParserNode &outer_structure_parser_node: basic_entry_points) {
        if (outer_structure_parser_node.as_structure_node() && !outer_structure_parser_node.as_structure()) {
            StructureNode &outer_structure_node = *outer_structure_parser_node.as_structure_node();
            Structure &inner_structure = outer_structure_node.ref_structure;
            // if (entry_point_parents.find(inner_structure) == entry_point_parents.end()) {
            //     entry_point_parents[inner_structure] = []
            ParserNode * par = outer_structure_node.parent;
            if (par == nullptr || par->as_structure()) {
                fprintf(stderr, "parent failed\n"); //: " + str(outer_structure_node))
                continue;
            }
            ParserStructureStack structure_stack(outer_structure_node.structure);
            // structure_stack.stack[:] = [outer_structure_node.structure]
            structure_stack.clear_ops();
            ParserNode * result_exp = par->next(NUL_CHR, &structure_stack, 0, true, &outer_structure_node);
            if (result_exp == nullptr) {
                fprintf(stderr, "par->next failed\n"); //: " + str(outer_structure_node))
                std::cout << "Add entry point parent " << inner_structure.name << ": NULL " << " from "
                          << outer_structure_node << std::endl;
                return;
            }
            // assert (result_exp is not nullptr)
            entry_point_parents[&inner_structure].emplace_back(
                    std::make_pair(std::reference_wrapper<ParserNode>(*result_exp),
                                   std::reference_wrapper<StructureNode>(outer_structure_node)));
//            entry_point_parents[&inner_structure].emplace_back(std::make_pair(std::reference_wrapper<ParserNode>(*result_exp), std::reference_wrapper<StructureNode>(outer_structure_node)));
//            std::cout << "Add entry point parent " << inner_structure.name << ": " << *result_exp << " from " << outer_structure_node << std::endl;
        }
    }

    // Key
    std::unordered_map<TokensCalculatedKey, std::unordered_set<CalculatedToken, TokensCalculatedKeyHasher, TokensCalculatedKeyEq>, TokensCalculatedKeyHasher, TokensCalculatedKeyEq> tokens_calculated;
    // : dict[tuple[SingleNodeKey, str], set[tuple[OpTuple, StructureNode, tuple[ParserNode, const char32_t*]]]] = {}
    // token_queue: list[tuple[ParserNode, const char32_t*]] = []
    // token_char_list: const char32_t*
    std::deque<std::pair<std::reference_wrapper<ParserNode>, const char32_t *>> token_queue;
    for (ParserNode &node: basic_entry_points) {
//        if (node.get_class_name() == EndNode) {
//            continue;
//        }
        if (VERBOSE >= 1) {
            std::cout << node << std::endl;
        }
        ParserStructureStack ts(node.structure);
        // Major optimization hack (24s -> 8s):
        // If the least common english letter matches, we try all possible strings.
        // If not, then we will only advance letter-by-letter, special words, or with non-alphabetic tokens.
        ParserNode * alpha_guess = node.n(U'z', &ts);
        (void)alpha_guess;
        for (std::pair<int, std::u32string> &id_str: stripped_tokens_to_process) {
            int token_id = id_str.first;
            std::u32string token_string = id_str.second;
            bool is_whitespace = (unsigned)std::count_if(token_string.begin(), token_string.end(), [](char32_t ch) { return std::isspace((int)ch); }) == (unsigned)token_string.size();
            (void)is_whitespace;
            const char32_t *token_cstr = token_string.c_str();
            (void)token_cstr;
            if (node.prefix_filter(token_cstr[0])) {
                //if ((token_cstr[0] != 0 && token_cstr[0] <= 0xff && token_cstr[1] == 0) || is_whitespace ||
                //token_string == U"true" || token_string == U"false" || token_string == U"null" ||
                //token_string.find_first_of(U"{}[]\":,\n-0123456789") != std::u32string::npos ||
                //(alpha_guess != nullptr && alpha_guess->as_structure_node() == nullptr)) {
                const char32_t *token_char_list = tokenizer_data.token_strings[token_id].c_str();
                tokens_calculated[std::make_pair(node.node_key, token_char_list)].clear();
                token_queue.emplace_back(std::reference_wrapper<ParserNode>(node), token_char_list);
                //std::cout << "    token_queue.emplace_back " << node << " [" << token_char_list << "]" << std::endl;
            }
        }
    }

    for (; !token_queue.empty(); token_queue.pop_front()) {
        ParserNode & node = token_queue.front().first;
        const char32_t *orig_token_char_list = token_queue.front().second;
//        if (std::memcmp(orig_token_char_list, "t\0\0\0r\0\0\0u\0\0\0e",13) == 0) {
//            printf("Is true!\n");
//        }
        const char32_t *token_char_list = orig_token_char_list;
        SingleNodeKey node_id = node.node_key;
        NodeKeyTuple orig_node_key;
        orig_node_key.emplace_back(node_id);
        Structure &root_structure = node.structure;
        ParserNode * new_node = &node;

        ParserStructureStack structure_stack(node.structure);
        structure_stack.clear_ops();

        int num_chars = 0;
        // assert (new_node is not nullptr)  # To help mypy
        if (!token_char_list) {
            new_node = root_structure.tick(&structure_stack, new_node, END_CHR);
        } else {
            if (new_node == root_structure.root_node.get() && orig_token_char_list[0] == ' '
                && orig_token_char_list[1] != '\0' && orig_token_char_list[1] != ' ') {
                token_char_list++; // root node consumes leading whitespace.
            }
            while (char32_t this_char = *token_char_list) {
                new_node = root_structure.tick(&structure_stack, new_node, this_char);
                if (new_node && new_node->as_structure() && structure_stack.stack.size() > 1) {
                    ParserNode *outer_structure_node = structure_stack.pop();
                    new_node = outer_structure_node->parent;
                    if (new_node == nullptr || new_node->as_structure()) {
                        new_node = nullptr;
                        fprintf(stderr, "parent failed\n");
                    }
                    if (new_node) {
                        new_node = new_node->next(NUL_CHR, &structure_stack, 0, true, outer_structure_node);
                    }
                    if (new_node) {
                        // Retry the current char without advancing.
                        continue;
                    }
                }
                if (new_node && new_node->as_structure() && structure_stack.stack.size() <= 1) {
                    break;
                }
                if (!new_node) {
                    // Illegal token
                    structure_stack.clear_ops();
                    if (VERBOSE >= 4) {
                        std::cout << " -" << node << " Illegal token at " << num_chars << " of "
                                  << orig_token_char_list << std::endl;
                    }
                    break;
                }
                num_chars += 1;
                token_char_list++;
            }
        }
        OpTuple new_ops;
        for (ParserStructureStack::StructOperation op: structure_stack.ops) {
            if (op.pushStruct) {
                // assert (len(op.node_key) == 3) // StructureNode node_key always has tuple (containing_struct, index, child_struct)
                new_ops.emplace_back(StructOperation::push(op.pushStruct->node_key));
            } else if (op.appendChar) {
                if (new_node) {
                    new_ops.emplace_back(StructOperation::append(op.appendChar));
                }
            } else {
                new_ops.emplace_back(StructOperation::pop());
            }
        }
        if (new_node && new_node->as_structure()) {
            decltype(entry_point_parents)
            ::iterator iter = entry_point_parents.find(new_node->as_structure());
            if (iter != entry_point_parents.end()) {
                if (VERBOSE >= 3) {
                    std::cout << "  " << node << " Got StructureClass " << new_node->as_structure()->name << " ops "
                              << new_ops << " at " << num_chars << " of " << orig_token_char_list << std::endl;
                }
                for (const std::pair<std::reference_wrapper<ParserNode>, std::reference_wrapper<StructureNode>> &parent_and_outer_token_node: iter->second) {
                    ParserNode & parent = parent_and_outer_token_node.first;
                    StructureNode &outer_token_node = parent_and_outer_token_node.second;
                    std::pair<SingleNodeKey, const char32_t *> new_key_id = std::make_pair(parent.node_key,
                                                                                           token_char_list);
                    if (tokens_calculated.find(new_key_id) == tokens_calculated.end()) {
                        token_queue.emplace_back(std::reference_wrapper<ParserNode>(parent), token_char_list);
                    }
                    tokens_calculated[new_key_id].insert(
                            CalculatedToken(new_ops, outer_token_node, node, orig_token_char_list));
                }
            }
        } else {
            if (new_node) {
                // assert (num_chars == len(ll_to_str(orig_token_char_list)))
                if (VERBOSE >= 2) {
                    std::cout << "<>" << node << " Got Node " << *new_node
                              << " ops " << new_ops << " at " << orig_token_char_list << std::endl;
                }
                // if (precalculated_raw_graph.find(orig_node_key) == precalculated_raw_graph.end()) {
                //     // assert (len(orig_node_key[0]) > 1 and type(orig_node_key[0][1]) is int)
                //     precalculated_raw_graph[orig_node_key] = {};
                // }
                precalculated_raw_graph.graph[orig_node_key].insert(
                        std::make_pair(orig_token_char_list, std::make_pair(new_ops, new_node->node_key)));
            }
        }
    }

    if (VERBOSE >= 1) {
        std::cout << "Done 1" << std::endl;
    }
//    PrecalculatedRawGraph::GraphType precalculated_raw_copy = precalculated_raw_graph.graph;
    std::vector<std::reference_wrapper<PrecalculatedRawGraph::GraphType::value_type>> precalculated_raw_copy(
            precalculated_raw_graph.graph.begin(), precalculated_raw_graph.graph.end());
    std::sort(precalculated_raw_copy.begin(), precalculated_raw_copy.end(),
              [](const auto &a, const auto &b) { return a.get().first < b.get().first; });

    std::deque<std::pair<OpTuple, std::pair<std::reference_wrapper<std::unordered_set<CalculatedToken, TokensCalculatedKeyHasher, TokensCalculatedKeyEq>>, NodeKeyTuple>>> recursion_queue;
//    for (const decltype(tokens_calculated)::value_type &calc_item : tokens_calculated) {
//        std::cout << "Calculated token: " << calc_item.first.first << "," << calc_item.first.second << std::endl;
//    }
    for (PrecalculatedRawGraph::GraphType::value_type &graph_value: precalculated_raw_copy) {
//        std::cout << "Secondloop: " << graph_value.first << " :" << std::endl;
        // for node_key_tuple, subdict in list(precalculated_raw_graph.items())[:]:
        const NodeKeyTuple &node_key_tuple = graph_value.first;

        std::vector<std::u32string> token_list;
        token_list.reserve(graph_value.second.size());
        for (const PrecalculatedRawGraph::PerStateTokenMap::value_type &token_data: graph_value.second) {
            token_list.emplace_back(token_data.first);
        }
        std::sort(token_list.begin(), token_list.end());
        //for (PrecalculatedRawGraph::PerStateTokenMap::value_type &state_value : graph_value.second) {
        for (const std::u32string &token_str: token_list) {
            const PrecalculatedRawGraph::PerStateTokenMap::mapped_type &state_value = graph_value.second[token_str];
            // token_str, (new_ops, new_node_key)
//            std::cout << "    " << token_str << " : " << state_value.first << "," << state_value.second << std::endl;
//            const std::u32string &token_str = state_value.first;
            const OpTuple &new_ops = state_value.first;
            const SingleNodeKey &new_node_key = state_value.second;
            TokensCalculatedKey this_key = std::make_pair(node_key_tuple[0], token_str.c_str());
            decltype(tokens_calculated)
            ::iterator tokens_calc_iter = tokens_calculated.find(this_key);
//            std::cout << "Lookup calculated: " << node_key_tuple[0] << " \"" << token_str << ((tokens_calculated.find(this_key) == tokens_calculated.end()) ? "\" skip" : "\" found") << std::endl;
            if (tokens_calculated.find(this_key) == tokens_calculated.end()) {
                continue;
            }
            recursion_queue.emplace_back(new_ops, std::make_pair(
                    std::reference_wrapper<std::unordered_set<CalculatedToken, TokensCalculatedKeyHasher, TokensCalculatedKeyEq>>(
                            tokens_calc_iter->second),
                    node_key_tuple));
            for (; !recursion_queue.empty(); recursion_queue.pop_front()) {
                const OpTuple &cur_ops = recursion_queue.front().first;
                std::unordered_set<CalculatedToken, TokensCalculatedKeyHasher, TokensCalculatedKeyEq> &list_of_children = recursion_queue.front().second.first;
                const NodeKeyTuple &node_list = recursion_queue.front().second.second;
                if (node_list.size() > 100) {
                    fprintf(stderr,
                            "Infinite loop detected\n"); // at " + str(recursion_queue[queue_idx]) + " | " + str((this_key, (new_ops, new_node_key))))
                    return;
                }
                for (const CalculatedToken &calculated_token: list_of_children) {
                    // for child_ops, outer_token_node, child_key in list_of_children:
                    NodeKeyTuple child_node_list = node_list;
                    child_node_list.pop_back();
                    child_node_list.emplace_back(calculated_token.node.node_key);
                    child_node_list.emplace_back(calculated_token.parser_node.node_key);

                    OpTuple total_ops = calculated_token.ops;
                    total_ops.emplace_back(StructOperation::pop());
                    total_ops.insert(total_ops.end(), cur_ops.begin(), cur_ops.end());
                    // if child_inner_key[0] not in precalculated_raw_graph:
                    //     precalculated_raw_graph[child_inner_key[0]] = {}
                    // assert (len(child_inner_key[0][0]) > 1 and type(child_inner_key[0][0][1]) is int)
                    precalculated_raw_graph.graph[child_node_list].insert(
                            std::make_pair(calculated_token.chars, std::make_pair(total_ops, new_node_key)));
                    if (VERBOSE >= 2) {
                        std::cout << "^^" << child_node_list << " Got Node " << new_node_key << " ops " << total_ops
                                  << " at "
                                  << calculated_token.chars << std::endl;
                    }
                    TokensCalculatedKey calc_key = std::make_pair(calculated_token.parser_node.node_key,
                                                                  calculated_token.chars);
                    decltype(tokens_calculated)
                    ::iterator calc_tokens_calc_iter = tokens_calculated.find(calc_key);
                    if (calc_tokens_calc_iter != tokens_calculated.end()) {
                        recursion_queue.emplace_back(total_ops, std::make_pair(
                                std::reference_wrapper<std::unordered_set<CalculatedToken, TokensCalculatedKeyHasher, TokensCalculatedKeyEq>>(
                                        calc_tokens_calc_iter->second),
                                child_node_list));
                    }
                }
            }
        }
    }
}


StructureOpGraph::StructureOpGraph(const PrecalculatedRawGraph &precalculated_raw_graph, bool structure_only,
                                   const TokenizerData &tokenizer_data)
        : precalculated_raw_graph(precalculated_raw_graph),
          tokenizer_data(tokenizer_data),
          token_strings(tokenizer_data.token_strings) {
    for (const PrecalculatedRawGraph::GraphType::value_type &raw_data: precalculated_raw_graph.graph) {
        const NodeKeyTuple &node_key = raw_data.first;
        const PrecalculatedRawGraph::PerStateTokenMap &subdict = raw_data.second;
        this->history_length = std::max((int) node_key.size(), this->history_length);
        std::shared_ptr<OpTupleToToken> self_current_ops = std::make_shared<OpTupleToToken>();
        std::shared_ptr<NodeKeyToToken> self_outgoing_no_op_states = std::make_shared<NodeKeyToToken>();
        for (const PrecalculatedRawGraph::PerStateTokenMap::value_type &output_data: subdict) {
            const std::u32string &tok_str = output_data.first;
            const OpTuple &new_ops = output_data.second.first;
            const SingleNodeKey &new_node_key = output_data.second.second;
            std::pair<TokenizerData::TokensToIdMap::const_iterator, TokenizerData::TokensToIdMap::const_iterator> it =
                tokenizer_data.tokens_to_id.equal_range(tok_str);
            // Some tokens get constructed as part of calculating a larger recursive token.
            // These tokens which are not in tokens_to_id will exist in precalculated_raw but must not be used.
            while (it.first != it.second) {
                int tok_id = (int)it.first->second;
                ++it.first;
                // if (structure_only) {
                //     new_ops = tuple(op for op in new_ops if not type(op) is str);
                // }
                if (new_ops.empty()) {
                    this->outgoing_no_op_states[node_key] = self_outgoing_no_op_states;
                    if (self_outgoing_no_op_states->find(new_node_key) == self_outgoing_no_op_states->end()) {
                        self_outgoing_no_op_states->insert(
                                NodeKeyToToken::value_type(new_node_key, TokenTensor(this->tokenizer_data)));
                    }
                    (*self_outgoing_no_op_states)[new_node_key]->add(tok_id);
                } else {
                    this->current_ops[node_key] = self_current_ops;
                    if (self_current_ops->find(new_ops) == self_current_ops->end()) {
                        self_current_ops->insert(OpTupleToToken::value_type(new_ops, TokenTensor(this->tokenizer_data)));
                    }
                    (*self_current_ops)[new_ops]->add(tok_id);
                }
            }
        }
    }
}

void StructureOpGraph::_explore_neighbors(const NodeKeyTuple &node_key, OpTupleToToken &neighbor_op_set,
                                          std::unordered_set<NodeKeyTuple, CustomHasher> &visited,
                                          TokenTensor parent_tok_ids) const {
    NodeKeyToToken no_op_set;
    for (unsigned i = 0; i < node_key.size(); i++) {
        NodeKeyTuple inner_node_key(node_key.begin() + i, node_key.end());
        decltype(outgoing_no_op_states)
        ::const_iterator niter = outgoing_no_op_states.find(inner_node_key);
        if (niter != outgoing_no_op_states.end()) {
            for (const NodeKeyToToken::value_type &nodekey_token_pair: *niter->second) {
                const CPUBitTensor &tok_ids = parent_tok_ids.get() ? *parent_tok_ids : *nodekey_token_pair.second;
                NodeKeyToToken::iterator iter = no_op_set.find(nodekey_token_pair.first);
                if (iter == no_op_set.end()) {
                    iter = no_op_set.insert(NodeKeyToToken::value_type(nodekey_token_pair.first,
                                                                       TokenTensor(this->tokenizer_data))).first;
                }
                iter->second->update(tok_ids);
            }
        }
        decltype(current_ops)
        ::const_iterator opiter = current_ops.find(inner_node_key);
        if (opiter != current_ops.end()) {
            for (const OpTupleToToken::value_type &op_token_pair: *opiter->second) {
                const CPUBitTensor &tok_ids = parent_tok_ids.get() ? *parent_tok_ids : *op_token_pair.second;
                OpTupleToToken::iterator iter = neighbor_op_set.find(op_token_pair.first);
                if (iter == neighbor_op_set.end()) {
                    iter = neighbor_op_set.insert(
                            OpTupleToToken::value_type(op_token_pair.first, TokenTensor(this->tokenizer_data))).first;
                }
                iter->second->update(tok_ids);
            }
        }
    }
    for (const NodeKeyToToken::value_type &nodekey_token_pair: no_op_set) {
        NodeKeyTuple neigh_node_key = node_key;
        const SingleNodeKey &neigh_single_node_key = nodekey_token_pair.first;
        const TokenTensor &tok_ids = nodekey_token_pair.second;
        neigh_node_key.back() = neigh_single_node_key;
        if (visited.find(neigh_node_key) == visited.end()) {
            visited.insert(neigh_node_key);
            this->_explore_neighbors(neigh_node_key, neighbor_op_set, visited, tok_ids);
            visited.erase(neigh_node_key);
        }
    }
}

template<class T>
static std::set<InternedString> get_structure_names(const T &structures, bool is_string) {
    std::set<InternedString> struct_names;
    for (const typename decltype(structures)::value_type & struct_pair : structures) {
        if (struct_pair.second->is_string == is_string) {
            struct_names.insert(struct_pair.second->name);
        }
    }
    return struct_names;
}

PrecalculatedStructureGraph::PrecalculatedStructureGraph(std::unique_ptr<TokenizerData> tokenizer_data_,
                                                         std::unique_ptr<NodeStructureGraph> node_structure_graph_)
        : tokenizer_data(std::move(tokenizer_data_)),
          structure_graph(std::move(node_structure_graph_)),
          intern_table(structure_graph->intern_table),
          token_strings(tokenizer_data->token_strings),
          tokens_to_id(tokenizer_data->tokens_to_id),
//        string_structures(get_structure_names(structure_graph->structures, true)),
//        recursive_structures(get_structure_names(structure_graph->structures, false)),
          root_node_key{structure_graph->root_structure, 1, InternedString()},
          precalculated_raw_graph(std::make_unique<PrecalculatedRawGraph>(*tokenizer_data, *structure_graph)),
//        op_struct_tok(std::make_unique<StructureOpGraph>(*precalculated_raw_graph, true, *tokenizer_data)),
//        op_graph_tok(precalculated_raw_graph, false, *tokenizer_data),
          history_length(1) {

    for (const PrecalculatedRawGraph::GraphType::value_type &kv: this->precalculated_raw_graph->graph) {
        if (kv.first.size() == 1 && kv.first[0].structName == root_node_key.structName &&
            kv.first[0].pos == root_node_key.pos) {
            this->root_node_key = kv.first[0];
            break;
        }
        const NodeKeyTuple &node_key = kv.first;
        this->history_length = std::max((unsigned)node_key.size(), this->history_length);
    }

    std::vector<TokenTensor> precalculated_vectors;

    TokenTensor end_tensor = TokenTensor(*tokenizer_data);
    for (unsigned tok = 0; tok < (unsigned) token_strings.size(); tok++) {
        if (token_strings[tok].empty()) {
            end_tensor->add(tok);
        }
    }
    NodeKeyTuple end_state_key = {{spaceStr, 0, InternedString()}};
    precalculated_state_keys.emplace_back(end_state_key);
    precalculated_vectors.emplace_back(end_tensor);
    TokenTensor space_tensor = TokenTensor(*tokenizer_data);
    std::set<std::pair<int, unsigned>> space_tokens;
    max_spaces = 0;
    for (unsigned tok = 0; tok < (unsigned) token_strings.size(); tok++) {
        if (!token_strings[tok].empty() && token_strings[tok].find_first_not_of(U' ') == std::u32string::npos) {
            space_tensor->add(tok);
            space_tokens.insert(std::make_pair(-(int) token_strings[tok].size(), tok));
            max_spaces = std::max(max_spaces, (unsigned) token_strings[tok].size());
        }
    }
    for (unsigned num_spaces = 1; num_spaces <= max_spaces; num_spaces++) {
        TokenTensor tensor = TokenTensor(*tokenizer_data);
        tensor->add(space_tokens.lower_bound(std::make_pair(-num_spaces, 0))->second);
        NodeKeyTuple state_key = {{spaceStr, (int) num_spaces, InternedString()}};
        precalculated_state_keys.emplace_back(state_key);
        precalculated_vectors.emplace_back(tensor);
    }

    InternedString GreedyConstantCharNode = structure_graph->intern_table.intern(U"GreedyConstantCharNode");

    std::vector<std::reference_wrapper<PrecalculatedRawGraph::GraphType::value_type>> sorted_data(
            precalculated_raw_graph->graph.begin(), precalculated_raw_graph->graph.end());
    std::sort(sorted_data.begin(), sorted_data.end(),
              [](const auto &a, const auto &b) { return a.get().first < b.get().first; });
    for (const PrecalculatedRawGraph::GraphType::value_type &kv: sorted_data) { //precalculated_raw_graph->graph) {
        const NodeKeyTuple &state_key = kv.first;
        TokenTensor tt = TokenTensor(*tokenizer_data);
        //std::cout << state_key << " =>" << std::endl;
        for (unsigned i = 1; i <= state_key.size(); i++) {
            PrecalculatedRawGraph::GraphType::const_iterator iter = precalculated_raw_graph->graph.find(
                    NodeKeyTuple(state_key.end() - i, state_key.end()));
            //std::cout << "  State key " << std::to_string(i) << ": " << NodeKeyTuple(state_key.end() - i, state_key.end()) << std::endl;
            if (iter != precalculated_raw_graph->graph.end()) {
                std::vector<std::u32string> token_list;
                token_list.reserve(iter->second.size());
                bool strip_middle_tokens = false;
                std::unordered_set<std::u32string> found_middle_tokens;
                for (const PrecalculatedRawGraph::PerStateTokenMap::value_type &token_data: iter->second)
                {
                    InternedString new_struct_name = token_data.second.second.nodeType;
                    if (new_struct_name == GreedyConstantCharNode)
                    {
                        strip_middle_tokens = true;
                        found_middle_tokens.emplace(token_data.first);
                    }
                }
                if (strip_middle_tokens) {
                    for (const PrecalculatedRawGraph::PerStateTokenMap::value_type &token_data: iter->second)
                    {
                        InternedString new_struct_name = token_data.second.second.nodeType;
                        if (new_struct_name != GreedyConstantCharNode)
                        {
                            std::u32string substring = token_data.first;
                            while (substring.size() > 1)
                            {
                                substring.pop_back();
                                found_middle_tokens.erase(substring);
                            }
                        }
                    }
                }
                for (const PrecalculatedRawGraph::PerStateTokenMap::value_type &token_data: iter->second) {
                    InternedString new_struct_name = token_data.second.second.nodeType;
                    if (new_struct_name == GreedyConstantCharNode)
                    {
                        if (found_middle_tokens.find(token_data.first) == found_middle_tokens.end()) {
                            continue; // There is a non-middle token reachable, so remove all middle tokens.
                        }
                    }
                    int push_count = 0;
                    int pop_count = 0;
                    int append_count = 0;
                    if (!structure_graph->structures[token_data.second.second.structName]->permit_hybrid_ops)
                    {
                        for (const StructOperation & op : token_data.second.first)
                        {
                            switch (op.type)
                            {
                            case StructOperation::APPEND_CHAR: append_count++; break;
                            case StructOperation::POP: pop_count++; break;
                            case StructOperation::PUSH: push_count++; break;
                            }
                        }
                        if ((push_count >= 1 || pop_count >= 1) && append_count >= 1)
                        {
                            // Forbid complex tokens which push more than once and also append characters.
                            // For example, forbid {". or ","false or ."} but permit "false or :- or false"
                            if (VERBOSE >= 2) {
                                std::cout << "Forbid hybrid ops " << token_data.first << " at " << state_key << std::endl;
                            }
                            continue;
                        }
                    }
                    token_list.emplace_back(token_data.first);
                }
                std::sort(token_list.begin(), token_list.end());
                for (const std::u32string &tok_str: token_list) { //iter->second) {
                    //const std::u32string &tok_str = token_data.first;
                    std::pair<TokenizerData::TokensToIdMap::const_iterator, TokenizerData::TokensToIdMap::const_iterator> itertok =
                        tokens_to_id.equal_range(tok_str);
                    while (itertok.first != itertok.second) {
                        unsigned tok_id = itertok.first->second;
                        ++itertok.first;
                        if ((state_key.back().nodeType == indentStr || state_key.back().nodeType == indentEndStr) &&
                            space_tensor->has(tok_id)) {
                            continue;
                        }
                        tt->add(tok_id);
                        if (VERBOSE >= 3) {
                            std::cout << "    Allowed token " << std::to_string(tok_id) << ": " << tokenizer_data->token_strings[tok_id] << std::endl;
                        }
                    }
                }
            }
        }
        precalculated_state_keys.emplace_back(state_key);
        precalculated_vectors.emplace_back(tt);
    }
    size_t token_size = tokenizer_data->token_strings.size();
    size_t num_vectors = precalculated_state_keys.size();
    precalculated_vectors_tensor.reset(new uint8_t[num_vectors * tokenizer_data->padded_token_size]);
    memset(precalculated_vectors_tensor.get(), 0, num_vectors * tokenizer_data->padded_token_size);
    for (size_t i = 0; i < num_vectors; ++i) {
        precalculated_indices_by_state_key[precalculated_state_keys[i]] = i;
        const CPUBitTensor &tensor_data = *precalculated_vectors[i];
        for (size_t token = 0; token < token_size; token++) {
            if (tensor_data.has(token)) {
                precalculated_vectors_tensor[i * tokenizer_data->padded_token_size + token] = 1;
            }
        }
    }
    for (unsigned num_spaces = 1; precalculated_indices_by_state_key.find({{spaceStr, (int) num_spaces, InternedString()}}) !=
                                  precalculated_indices_by_state_key.end(); num_spaces++) {
        max_spaces = num_spaces;
    }
    for (const PrecalculatedRawGraph::GraphType::value_type &kv: this->precalculated_raw_graph->graph) {
        for (const PrecalculatedRawGraph::PerStateTokenMap::value_type &tok_kv: kv.second) {
            std::pair<TokenizerData::TokensToIdMap::const_iterator, TokenizerData::TokensToIdMap::const_iterator> itertok =
                tokens_to_id.equal_range(tok_kv.first);
            while (itertok.first != itertok.second) {
                token_output_graph.insert(GraphType::value_type(std::make_pair(kv.first, itertok.first->second), tok_kv.second));
                ++itertok.first;
            }
        }
    }
}

PrecalculatedStructureGraph::PrecalculatedStructureGraph(std::unique_ptr<TokenizerData> tokenizer_data_,
                                                         std::unique_ptr<InternTable> intern_table_,
        //const std::set<InternedString> &string_structures, const std::set<InternedString> &recursive_structures,
                                                         SingleNodeKey root_node_key, unsigned history_length,
                                                         std::vector<NodeKeyTuple> &&precalculated_state_keys_,
                                                         std::unique_ptr<uint8_t[]> &&precalculated_vectors_tensor_,
                                                         const std::vector<PerTokenDataPair> &token_output_graph_)
        : tokenizer_data(std::move(tokenizer_data_)),
          intern_table_holder(std::move(intern_table_)),
          intern_table(*intern_table_holder),
          token_strings(tokenizer_data->token_strings),
          tokens_to_id(tokenizer_data->tokens_to_id),
//          string_structures(string_structures),
//          recursive_structures(recursive_structures),
          root_node_key(root_node_key),
          history_length(history_length),
          precalculated_state_keys(std::move(precalculated_state_keys_)),
          precalculated_vectors_tensor(std::move(precalculated_vectors_tensor_)),
          token_output_graph(token_output_graph_.begin(), token_output_graph_.end()) {

    for (unsigned int sz = precalculated_state_keys.size(), i = 0; i < sz; ++i) {
        precalculated_indices_by_state_key[precalculated_state_keys[i]] = i;
    }
    for (unsigned num_spaces = 1; precalculated_indices_by_state_key.find({{spaceStr, (int) num_spaces, InternedString()}}) !=
                                  precalculated_indices_by_state_key.end(); num_spaces++) {
        max_spaces = num_spaces;
    }
}

}

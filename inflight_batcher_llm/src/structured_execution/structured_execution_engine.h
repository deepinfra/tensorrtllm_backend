/* structure_execution_engine.h
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

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <map>
#include <set>
#include <cctype>
#include <locale>
#include <memory>

namespace structured_execution {

inline std::u32string to_ascii_u32string(const std::string &s) {
    return std::u32string(s.begin(), s.end());
}

template <class T>
inline std::u32string to_u32string(const T &data) {
    std::string s = std::to_string(data);
    return to_ascii_u32string(s);
}

struct TokenizerData;

template <typename ...Args>
struct PrintArgs {
    void operator()(Args... args) {
    }
};

template <typename ...Args>
void print_args(Args... args) {
    PrintArgs<Args...>()(args...);
}

template <typename ...Args>
struct LoggerInt;

template <typename ...Args>
struct Logger {
    LoggerInt<Args...> logger_int;
    Logger(Args... args)
            : logger_int(args...) {
        print_args(args...);
        std::cout << std::endl;
    }
};

template <typename T, typename ...Args>
struct LoggerInt<T, Args...> {
    template <typename... SubArg> friend struct Logger;
//private:
    LoggerInt<Args...> sub_logger;
    T t;
    LoggerInt(const T&t, Args... sub_logger_args)
        :sub_logger(sub_logger_args...), t(t) {
    }
    ~LoggerInt() {
        std::cout << " <- " << t;
    }
};

template <>
struct LoggerInt<> {
    template <typename... SubArg> friend struct Logger;
//private:
    ~LoggerInt() {
        std::cout << std::endl;
    }
};

struct CPUBitTensor {
    struct alignas(64) BitChunk {
        uint64_t bits[8];
    };

    std::vector<BitChunk> bit_vector;


    bool has(unsigned bit) const {
        return (bit >> 9) < bit_vector.size() && (bit_vector[bit >> 9].bits[(bit >> 6) & 7] & (1ULL << (bit & 63))) != 0;
    }

    void add(unsigned bit) {
        if ((bit >> 9) < bit_vector.size()) {
            bit_vector[bit >> 9].bits[(bit >> 6) & 7] |= (1ULL << (bit & 63));
        }
    }

    void remove(unsigned bit) {
        if ((bit >> 9) < bit_vector.size()) {
            bit_vector[bit >> 9].bits[(bit >> 6) & 7] &= ~(1ULL << (bit & 63));
        }
    }

    void update(const CPUBitTensor &other) {
        for (unsigned i = 0; i < other.bit_vector.size(); i++) {
            for (unsigned j = 0; j < 8; j++) {
                bit_vector[i].bits[j] |= other.bit_vector[i].bits[j];
            }
        }
    }

    void difference_update(const CPUBitTensor &other) {
        for (unsigned i = 0; i < other.bit_vector.size(); i++) {
            for (unsigned j = 0; j < 8; j++) {
                bit_vector[i].bits[j] &= ~other.bit_vector[i].bits[j];
            }
        }
    }

    void intersection_update(const CPUBitTensor &other) {
        for (unsigned i = 0; i < other.bit_vector.size(); i++) {
            for (unsigned j = 0; j < 8; j++) {
                bit_vector[i].bits[j] &= other.bit_vector[i].bits[j];
            }
        }
    }
    CPUBitTensor(const TokenizerData &tokenizer_data);
};
typedef std::shared_ptr<CPUBitTensor> TokenTensor;


// const int VERBOSE = 0;


struct TokenizerData {
    std::vector<std::u32string> token_strings;
    typedef std::unordered_multimap<std::u32string, unsigned> TokensToIdMap;
    TokensToIdMap tokens_to_id;
    size_t padded_token_size;
    int bit_tensor_size;

    explicit TokenizerData(std::vector<std::u32string> &token_strings)
        : token_strings(token_strings), padded_token_size((token_strings.size() + 127) & (~127)) {
        for (unsigned i = 0; i < token_strings.size(); i++) {
            tokens_to_id.emplace(token_strings[i], i);
        }
        bit_tensor_size = (int)((token_strings.size() + 511) >> 9);
    }
    TokenizerData(const TokenizerData &other) = delete;
    TokenizerData &operator=(const TokenizerData &other) = delete;

    explicit operator TokenTensor() const {
        return std::make_shared<CPUBitTensor>(*this);
    }
};

inline CPUBitTensor::CPUBitTensor(const TokenizerData &tokenizer_data)
    : bit_vector(tokenizer_data.bit_tensor_size) {
}


constexpr char32_t END_CHR = (char32_t)-1;
constexpr char32_t NUL_CHR = (char32_t)0;

class InternTable;
class ParserNode;

struct InternedString {
    static const InternedString EMPTY;
    unsigned int stringId = 0;
    const std::u32string &get(const InternTable &internTable) const;

    bool operator==(const InternedString &oth) const {
        return stringId == oth.stringId;
    }
    bool operator<(const InternedString &oth) const {
        return stringId < oth.stringId;
    }
    bool operator!=(const InternedString &oth) const {
        return stringId != oth.stringId;
    }
    operator bool() const {
        return stringId != 0;
    }
    bool operator!() const {
        return stringId == 0;
    }
};

inline const InternedString InternedString::EMPTY = {};

struct InternedStringHash {
    std::size_t operator()(const InternedString &arg) const {
        return std::hash<int>{}((int) arg.stringId);
    }
};

class InternTable {
public:
    /*
    struct StringRefHolder {
        const std::u32string &data;

        StringRefHolder(const std::u32string & ref)
            : data(ref) {
        }
        operator const std::u32string &() const {
            return data;
        }
        bool operator==(const StringRefHolder &other) const {
            return other.data == data;
        }
        bool operator<(const StringRefHolder &other) const {
            return other.data < data;
        }
    };
     */
    std::vector<std::u32string> table;
    std::unordered_map<std::u32string, unsigned int> string_to_index;
    InternedString intern(const std::u32string &str) {
        std::unordered_map<std::u32string, unsigned int>::const_iterator iter = string_to_index.find(str);
        if (iter == string_to_index.end()) {
            unsigned int idx = table.size();
            table.emplace_back(str);
            string_to_index.insert(std::make_pair(str, idx));
            return { idx };
        }
        return { iter->second };
    }
    InternedString lookup(const std::u32string &str) const {
        std::unordered_map<std::u32string, unsigned int>::const_iterator iter = string_to_index.find(str);
        if (iter == string_to_index.end()) {
            return InternedString();
        }
        return { iter->second };
    }
    const std::u32string &get(InternedString str) const {
        return table[str.stringId];
    }
    InternTable() {
        intern(U"");
    }
    InternTable(const InternTable&) = delete;
    InternTable &operator=(const InternTable&) = delete;
};

inline const std::u32string &InternedString::get(const InternTable &internTable) const {
    return internTable.table[stringId];
}



struct SingleNodeKey {
    InternedString structName;
    int pos = 0;
    InternedString nodeType;
    static const SingleNodeKey null() {
        return {InternedString::EMPTY, 0, InternedString::EMPTY};
    }
    bool operator == (const SingleNodeKey &oth) const {
        return structName == oth.structName && pos == oth.pos && nodeType == oth.nodeType;
    }
    bool operator != (const SingleNodeKey &oth) const {
        return ! (*this == oth);
    }
    bool operator <(const SingleNodeKey &oth) const {
        if (structName == oth.structName) {
            if (pos == oth.pos) {
                return nodeType < oth.nodeType;
            }
            return pos < oth.pos;
        }
        return structName < oth.structName;
    }
    std::u32string to_string(const InternTable &internTable) const {
        return internTable.get(structName) + U"," + internTable.get(nodeType);
    }
};
typedef SingleNodeKey StructureNodeKey;
typedef std::vector<SingleNodeKey> NodeKeyTuple;

struct StructOperation {
    enum Type { PUSH, POP, APPEND_CHAR} type = POP;
    char32_t appendChar = 0;
    StructureNodeKey pushStruct;
    static StructOperation append(char32_t chr) {
        return {APPEND_CHAR, chr, SingleNodeKey::null()};
    }
    static StructOperation push(StructureNodeKey node) {
        return {PUSH, 0, node};
    }
    static StructOperation pop() {
        return {POP, 0, SingleNodeKey::null()};
    }
    bool operator==(const StructOperation &oth) const {
        return type == oth.type && appendChar == oth.appendChar && pushStruct == oth.pushStruct;
    }
    bool operator<(const StructOperation &oth) const {
        if (type != oth.type) {
            return (int)type < (int)oth.type;
        }
        if (type == PUSH) {
            return pushStruct < oth.pushStruct;
        }
        if (type == APPEND_CHAR) {
            return appendChar < oth.appendChar;
        }
        return false;
    }
};

typedef std::vector<StructOperation> OpTuple;

struct CustomHasher {
    std::size_t operator()(const InternedString &arg) const {
        return InternedStringHash()(arg);
    }
    std::size_t operator()(const SingleNodeKey &arg) const {
        return (*this)(arg.structName) * 7 + (*this)(arg.nodeType) * 11 + std::hash<int>{}(arg.pos) * 13;
    }
    std::size_t operator()(const NodeKeyTuple &arg) const {
        std::size_t h = arg.size();
        for(const SingleNodeKey& el : arg) {
            h ^= (*this)(el) + 0x9e3779b9 * ((h << 6) + (h >> 2));
        }
        return h;
    }
    std::size_t operator()(const std::pair<NodeKeyTuple, unsigned> &arg) const {
        std::size_t h = arg.first.size();
        for(const SingleNodeKey& el : arg.first) {
            h ^= (*this)(el) + 0x9e3779b9 * ((h << 6) + (h >> 2));
        }
        h ^= arg.second + 0x9e3779b9 * ((h << 6) + (h >> 2));
        return h;
    }
    std::size_t operator()(const StructOperation &arg) const {
        return std::hash<size_t>{}(((*this)(arg.pushStruct) + arg.appendChar) * 3 + ((int)arg.type));
    }
    std::size_t operator()(const OpTuple &arg) const {
        std::size_t h = arg.size();
        for(const StructOperation& el : arg) {
            h ^= (*this)(el) + 0x9e3779b9 * ((h << 6) + (h >> 2));
        }
        return h;
    }
};

class StructureNode;
class Sequence;
class Structure;
class NodeStructureGraph;
class ParserStructureStack;

struct PrecalculatedRawGraph{
    typedef std::pair<OpTuple, SingleNodeKey> PerTokenOutput;
    typedef std::unordered_map<std::u32string, const PerTokenOutput> PerStateTokenMap;
    typedef std::unordered_map<NodeKeyTuple, PerStateTokenMap, CustomHasher> GraphType;
    GraphType graph;

    PrecalculatedRawGraph(const TokenizerData &tokenizer_data, const NodeStructureGraph &node_graph);

    PrecalculatedRawGraph(const PrecalculatedRawGraph&) = delete;
    PrecalculatedRawGraph &operator=(const PrecalculatedRawGraph*) = delete;
};


typedef std::unordered_map<OpTuple, TokenTensor, CustomHasher> OpTupleToToken;
typedef std::unordered_map<SingleNodeKey, TokenTensor, CustomHasher> NodeKeyToToken;

#define PARSER_NODE_CLASS(x) const char32_t *generate_class_name() const override { return U ## #x; }


class ParserNode {
    // def __repr__(self) -> str:
    //     return ('<%s(%s)%r>' % (this->__class__.__name__, limit_str_length(repr(this->expr), 60), this->path) +
    //             "@" + this->structure.name + "/" + str(this->structure.node_to_index.get(self, nullptr)))
    mutable InternedString class_name;

protected:

    virtual const char32_t *generate_class_name() const = 0;

public:
    Structure &structure;
    SingleNodeKey node_key;

    ParserNode *parent;
    int index = -1;

    InternedString path_key;
    int path_key_idx = 0;
    bool reachable = false;
    bool hidden_output = false;
    std::u32string path;

    explicit ParserNode(Structure &structure);
    virtual ~ParserNode() = default;

    virtual SingleNodeKey generate_node_key();

    ParserNode *n(char32_t chr, ParserStructureStack *stack);

    InternedString get_class_name() const;

    virtual std::u32string get_debug_name() const;

    virtual int virtual_which_match(char32_t chr) = 0;

    virtual ParserNode *virtual_next(char32_t chr, ParserStructureStack *stack, int which_match, bool success, ParserNode *from_child);

    inline ParserNode *next(char32_t chr, ParserStructureStack *stack, int which_match, bool success = false, ParserNode *from_child = nullptr) {
        ParserNode *ret = virtual_next(chr, stack, which_match, success, from_child);
        return ret;
    }

    inline int which_match(char32_t chr) {
        int ret = virtual_which_match(chr);
//        Logger logger(ret, chr);
        return ret;
    }

    Structure *as_structure() const;

    virtual StructureNode *as_structure_node() {
        return nullptr;
    }
    virtual Sequence *as_sequence_node() {
        return nullptr;
    }
    virtual bool is_optional_root() const {
        return false;
    }
    virtual bool prefix_filter(char32_t chr) const {
        return true;
    }
    virtual unsigned int get_child_count() const {
        return 0;
    }
    virtual ParserNode *get_child(unsigned int idx) const {
        return nullptr;
    }
};

class StructureNode : public ParserNode { // ConstantNode
    PARSER_NODE_CLASS(StructureNode);
public:
    Structure &ref_structure;

    StructureNode(Structure &parent_structure, StructureNode &ref_structure);

    SingleNodeKey generate_node_key() override;

    // return this->expr.name + "@" + this->structure.name + "/" + str(this->structure.node_to_index.get(self, nullptr))

    int virtual_which_match(char32_t chr) override;

    ParserNode *virtual_next(char32_t chr, ParserStructureStack *stack, int which_match, bool success, ParserNode *from_child) override;

    StructureNode *as_structure_node() override {
        return this;
    }
};

class EndNode : public ParserNode {
    PARSER_NODE_CLASS(EndNode);
public:
    EndNode(Structure &structure) : ParserNode(structure) {}
    SingleNodeKey generate_node_key() override;

    int virtual_which_match(char32_t chr) override {
        if (chr == '\0') {
//            Logger log(2, chr);
            return 2;
        }
//        Logger log(-1, chr);
        return -1;
    }

    ParserNode *virtual_next(char32_t chr, ParserStructureStack *stack, int which_match, bool success = false, ParserNode *from_child = nullptr) override {
//        Logger log(parent, chr, stack, which_match, success, from_child);
        return parent;
    }
};

class NodeStructureGraph;

class Structure : public StructureNode {
    PARSER_NODE_CLASS(Structure);
    friend NodeStructureGraph;

    InternedString root_string;
    InternedString end_string;
    std::unique_ptr<EndNode> end_structure;
    std::vector<std::reference_wrapper<ParserNode>> indices;
//    std::unordered_map<ParserNode*, int> node_to_index;

    void _construct_indices_recursive(ParserNode *node, ParserNode *optional_root = nullptr, const std::u32string &path = std::u32string());

public:
    std::unique_ptr<ParserNode> root_node;
    NodeStructureGraph &graph;
    InternedString name;
    bool is_string = true;
    bool permit_hybrid_ops = true;

    Structure(NodeStructureGraph &graph, InternedString name);

    SingleNodeKey generate_node_key() override;

    InternedString intern_string(const std::u32string &str) const;
    const std::u32string &get_string(InternedString str) const;

    void construct_indices();

    int virtual_which_match(char32_t chr) override {
        if (chr == END_CHR) {
            return 1;
        }
        return 0;
    }

    ParserNode *virtual_next(char32_t chr, ParserStructureStack *stack, int which_match, bool success, ParserNode *from_child) override;

    ParserNode *tick(ParserStructureStack *structure_stack, ParserNode *exp_in, char32_t c);

};

class ParserStructureStack {
public:
    struct StructOperation {
        enum Type {
            PUSH,
            POP,
            APPEND_CHAR,
            SUCCESS
        } type;

        char32_t appendChar;
        StructureNode *pushStruct;

        static StructOperation append(char32_t chr) {
            return {APPEND_CHAR, chr, nullptr};
        }

        static StructOperation push(StructureNode *node) {
            return {PUSH, 0, node};
        }

        static StructOperation pop() {
            return {POP, 0, nullptr};
        }

        static StructOperation success() {
            return {SUCCESS, 0, nullptr};
        }
    private:
        StructOperation() = delete;
    };

    std::vector<StructOperation> ops;
    std::vector<std::reference_wrapper<StructureNode>> stack;

    ParserStructureStack(Structure &root_structure) {
        ops.emplace_back(StructOperation::push(&root_structure));
        stack.emplace_back(root_structure);
    }

    void append_char(char32_t chr) {
        if (!stack.empty()) {
            StructureNode &back_node = stack.back();
            if (back_node.ref_structure.is_string) {
                ops.emplace_back(StructOperation::append(chr));
            }
        }
    }

    void success() {
        ops.emplace_back(StructOperation::success());
    }

    void push(StructureNode *structure_node) {
        // assert(this->stack.empty() || !this->stack[-1]->structure_ref.is_string);
        ops.emplace_back(StructOperation::push(structure_node));
        stack.emplace_back(*structure_node);
    }

    StructureNode *pop() {
        this->ops.emplace_back(StructOperation::pop());
        StructureNode *ret = nullptr;
        if (this->stack.empty()) {
            fprintf(stderr, "pop from empty stack\n");
        } else {
            ret = &(StructureNode&)stack.back();
        }
        stack.pop_back();
        return ret;
    }

    void clear_ops() {
        ops.clear();
    }
};


inline ParserNode::ParserNode(Structure &structure)
    : structure(structure), parent(&structure) {
    index = (structure.index++);
}

inline SingleNodeKey ParserNode::generate_node_key() {
    return {structure.name, index, InternedString()};
}

inline InternedString ParserNode::get_class_name() const {
    if (class_name == InternedString()) {
        class_name = structure.intern_string(generate_class_name());
    }
    return class_name;
}

inline Structure *ParserNode::as_structure() const {
    if (this == &structure) {
        return &structure;
    }
    return nullptr;
}



inline StructureNode::StructureNode(Structure &parent_structure, StructureNode &ref_structure)
    : ParserNode(parent_structure),
      ref_structure(static_cast<Structure&>(ref_structure)) {
}

inline SingleNodeKey StructureNode::generate_node_key() {
    return {this->structure.name, this->index, ref_structure.name};
}

inline SingleNodeKey Structure::generate_node_key() {
    return {this->structure.name, this->index, root_string};
}

inline SingleNodeKey EndNode::generate_node_key() {
return {this->structure.name, this->index, get_class_name()};
}

inline std::u32string ParserNode::get_debug_name() const {
    return structure.get_string(get_class_name());
}

inline ParserNode *Structure::virtual_next(char32_t chr, ParserStructureStack *stack, int which_match, bool success, ParserNode *from_child) {
    // Structure::next will not automatically pop from the ParserStructureStack
    // if as_structure_node(), it is the caller's responsibility to get pop from the stack, perform next and rerun the tick:
    // outer_struct_node = stack.pop(); starting_node = outer_struct_node->parent->next(NUL_CHR, stack, 0, true, outer_structure_node);
    if (which_match == 1) {
        return nullptr;
    }
    if (success && from_child && from_child->path_key == root_string) {
        return end_structure.get();
    }
    return nullptr;
}

inline ParserNode *Structure::tick(ParserStructureStack *structure_stack, ParserNode *exp_in, char32_t c) {
    if (exp_in == this) {
        return nullptr;
    }
    ParserNode *exp = exp_in->n(c, structure_stack);
    if (exp == nullptr) {
        if (c == END_CHR && structure_stack) {
            structure_stack->success();
        }
        return nullptr;
    } else {
        // A grammar is not supposed to have a specific character that forces the parser into a StructureNode.
        // Here are some reasons for this design limitation:
        // 1. There would be no way to distinguish entering into a StructureNode and exiting from one.
        // 2. We never traverse on '': next(c='', success=true)
        // 3. It is undesirable to stop before we have pushed the StructureNode onto the structure_stack ops
        // (since we rely on ops in StructureOpGraph to select the correct token greedily)
        if (c == END_CHR && exp != this) {
            std::cerr << "Hit end chr early maybe" << std::endl;
            //print("c: " + str(c) + " exp: " + str(exp) + " self: " + str(self));
        }
        // assert (c != '' or exp == self)
        if (c == END_CHR && exp != this) {
            std::cerr << "Should be at this at end." << std::endl;
        }
        return exp;
    }
}

inline ParserNode * ParserNode::virtual_next(char32_t chr, ParserStructureStack *stack, int which_match, bool success, ParserNode *from_child) {
    if (stack && which_match != 0) {
        if (!hidden_output) {
            stack->append_char(chr);
        }
    }
    if (parent) {
        return parent->next(NUL_CHR, stack, 0, success || which_match != 0, this);
    }
    return nullptr;
}

inline int StructureNode::virtual_which_match(char32_t chr) {
    return ref_structure.root_node->which_match(chr);
}

inline ParserNode *StructureNode::virtual_next(char32_t chr, ParserStructureStack *stack, int which_match, bool success, ParserNode *from_child) {
    if (which_match != 0) {
        if (stack) {
            stack->push(this);
            return ref_structure.root_node->next(chr, stack, which_match);
        }
        return this;
    }
    return nullptr;
}


class WrapperNode : public ParserNode {
    PARSER_NODE_CLASS(WrapperNode);
protected:
    std::unique_ptr<ParserNode> only_child;
public:
    WrapperNode(Structure &structure, std::unique_ptr<ParserNode> &&only_child)
            : ParserNode(structure), only_child(std::move(only_child)) {
    }
    ParserNode *get_child(unsigned int idx) const override {
        return only_child.get();
    }
};


class ConstantCharNode : public ParserNode {
    PARSER_NODE_CLASS(ConstantCharNode);
    char32_t expr_min;
    char32_t expr_max_incl;
public:
    ConstantCharNode(Structure &structure, char32_t expr_min, char32_t expr_max_incl = 0)
        : ParserNode(structure), expr_min(expr_min), expr_max_incl(expr_max_incl == 0 ? expr_min : expr_max_incl) {
    }

    // def __repr__(self) -> str:
    //     return repr(this->expr) + "@" + this->structure.name + "/" + str(this->structure.node_to_index.get(self, nullptr))

    bool prefix_filter(char32_t chr) const override {
        return chr >= expr_min && chr <= expr_max_incl;
    }

    int virtual_which_match(char32_t chr) override {
        // if type(this->expr) is str:
        return chr >= expr_min && chr <= expr_max_incl ? 1 : 0;
        // if type(this->expr) is range:
        //     return 1 if (0 if chr == '' else ord(chr)) in this->expr else 0
        // if type(this->expr) is tuple:
        //     return 1 if chr in this->expr else 0
        // return 0
    }

    std::u32string get_debug_name() const override {
        if (expr_min == expr_max_incl) {
            char32_t ret[4] = {U'\'', expr_min, U'\'', 0};
            return ret;
        } else {
            char32_t ret[9] = {U'\'', expr_min, U'\'', U'.', U'.', U'\'', expr_max_incl, U'\'', 0};
            return ret;
        }
    }
};

class HiddenConstantCharNode : public ConstantCharNode {
    PARSER_NODE_CLASS(HiddenConstantCharNode);

public:
    HiddenConstantCharNode(Structure &structure, char32_t expr_min, char32_t expr_max_incl = 0)
    : ConstantCharNode(structure, expr_min, expr_max_incl) {
        hidden_output = true;
    }
};

class GreedyConstantCharNode : public ConstantCharNode {
    PARSER_NODE_CLASS(GreedyConstantCharNode);
    SingleNodeKey generate_node_key() override {
        return {this->structure.name, this->index, get_class_name()};
    }

public:
    GreedyConstantCharNode(Structure &structure, char32_t expr_min, char32_t expr_max_incl = 0)
        : ConstantCharNode(structure, expr_min, expr_max_incl) {
        hidden_output = true;
    }
};

class HiddenGreedyConstantCharNode : public GreedyConstantCharNode {
    PARSER_NODE_CLASS(HiddenGreedyConstantCharNode);

public:
    HiddenGreedyConstantCharNode(Structure &structure, char32_t expr_min, char32_t expr_max_incl = 0)
        : GreedyConstantCharNode(structure, expr_min, expr_max_incl) {
        hidden_output = true;
    }
};

class AnyOfCharNode : public ParserNode {
    PARSER_NODE_CLASS(AnyOfCharNode);
    std::u32string expr;
public:
    AnyOfCharNode(Structure &structure, std::u32string expr)
        : ParserNode(structure), expr(std::move(expr)) {
    }

    bool prefix_filter(char32_t chr) const override {
        return expr.find(chr) != std::u32string::npos;
    }

    int virtual_which_match(char32_t chr) override {
        return expr.find(chr) != std::u32string::npos;
    }

    std::u32string get_debug_name() const override {
        return U"AnyOfCharNode(/[" + expr + U"]/)";
    }
};

typedef std::vector<std::unique_ptr<ParserNode>> NodeList;


class Sequence : public ParserNode {
    PARSER_NODE_CLASS(Sequence);

protected:
    NodeList children;

public:
    Sequence(Structure &structure, NodeList &&children)
        : ParserNode(structure), children(std::move(children)) {
    }

    int virtual_which_match(char32_t chr) override {
        return children[0]->which_match(chr);
    }

    bool prefix_filter(char32_t chr) const override {
        return children[0]->prefix_filter(chr);
    }

    ParserNode * virtual_next(char32_t chr, ParserStructureStack *stack, int which_match, bool success, ParserNode *from_child) override {
        if (which_match != 0) {
            return children[0]->next(chr, stack, which_match);
        }
        if (!success) {
            return nullptr;
        }
        int last_finished_idx = -1;
        if (from_child && from_child->path_key_idx != -1) {
            last_finished_idx = from_child->path_key_idx;
        }
        unsigned int new_idx = (unsigned int)(last_finished_idx + 1);
        if (new_idx >= children.size() || !success) {
            return ParserNode::virtual_next(NUL_CHR, stack, 0, success, this);
        }
        return get_child(new_idx);
    }

    Sequence *as_sequence_node() override {
        return this;
    }
    ParserNode *get_child(unsigned int idx) const override {
        if (idx >= children.size()) {
            return nullptr;
        }
        return children[idx].get();
    }
    unsigned int get_child_count() const override {
        return children.size();
    }
};


class AnyOf : public ParserNode {
    PARSER_NODE_CLASS(AnyOf);
protected:
    NodeList children;

public:
    AnyOf(Structure &structure, NodeList &&children)
        : ParserNode(structure), children(std::move(children)) {
    }

    bool prefix_filter(char32_t chr) const override {
        for (unsigned int i = 0; i < children.size(); i++) {
            if (children[i]->prefix_filter(chr)) {
                return true;
            }
        }
        return false;
    }

    int virtual_which_match(char32_t chr) override {
        for (unsigned int i = 0; i < children.size(); i++) {
            if (children[i]->which_match(chr) > 0) {
                return i + 1;
            }
        }
        return 0;
    }

    ParserNode * virtual_next(char32_t chr, ParserStructureStack *stack, int which_match, bool success, ParserNode *from_child) override {
        if (which_match == 0) {
            if (!success) {
                return nullptr;
            }
            return ParserNode::virtual_next(NUL_CHR, stack, 0, success, this);
        } else {
            // assert (chr != NUL_CHR)
            ParserNode *next_node = children[which_match - 1].get();
            which_match = next_node->which_match(chr);
            return next_node->next(chr, stack, which_match);
        }
    }

    bool is_optional_root() const override {
        return true;
    }
    ParserNode *get_child(unsigned int idx) const override {
        if (idx >= children.size()) {
            return nullptr;
        }
        return children[idx].get();
    }
    unsigned int get_child_count() const override {
        return children.size();
    }
};


class ZeroOrMore : public WrapperNode {
    PARSER_NODE_CLASS(ZeroOrMore);
protected:
    bool only_once;
public:
    ZeroOrMore(Structure &structure, std::unique_ptr<ParserNode> &&only_child, bool only_once = false)
        : WrapperNode(structure, std::move(only_child)), only_once(only_once) {
    }

    int virtual_which_match(char32_t chr) override {
        if (chr == '\0') {
            return 2;
        }
        if (only_child->which_match(chr) != 0) {
            return 1;
        }
        ParserNode *nxt = ParserNode::virtual_next(NUL_CHR, nullptr, 0, true, this);
        if (nxt != nullptr and nxt->which_match(chr) != 0) {
            return 2;
        }
        return -1;
    }

    ParserNode *virtual_next(char32_t chr, ParserStructureStack *stack, int which_match, bool success, ParserNode *from_child) override {
        if (which_match == 1) {
            return only_child->next(chr, stack, only_child->which_match(chr));
        }
        if (which_match == -1 || which_match == 2) {
            //assert from_child == '' and chr != NUL_CHR
            ParserNode *next_node = ParserNode::virtual_next(NUL_CHR, stack, 0, true, this);
            if (next_node == nullptr) {
                return ParserNode::virtual_next(NUL_CHR, stack, 0, false, this);
            }
            if (from_child) {
                return next_node;
            }
            if (next_node->as_structure_node()) {
                return next_node;
            }
            which_match = next_node->which_match(chr);
            return next_node->next(chr, stack, which_match);
        } else if (success && !this->only_once) {
            // assert (from_child == this->__class__.__name__)
            return this;
        } else {
            // assert (from_child == this->__class__.__name__)
            return ParserNode::virtual_next(NUL_CHR, stack, 0, success, this);
        }
    }

    bool is_optional_root() const override {
        return true;
    }

    std::u32string get_debug_name() const override {
        return structure.get_string(get_class_name()) + U"[" + only_child->get_debug_name() + U"]";
    }
};

// Ensuring the correct amount of whitespace must be handled by the implementation.
class AutoIndent : public ZeroOrMore {
    PARSER_NODE_CLASS(AutoIndent);
    SingleNodeKey generate_node_key() override {
        return {this->structure.name, this->index, get_class_name()};
    }
public:
    AutoIndent(Structure &structure, std::unique_ptr<ParserNode> &&only_child)
        : ZeroOrMore(structure, std::move(only_child)) {
    }
};

// Ensuring the correct amount of whitespace must be handled by the implementation.
class AutoIndentEnd : public AutoIndent {
    PARSER_NODE_CLASS(AutoIndentEnd);
    SingleNodeKey generate_node_key() override {
        return {this->structure.name, this->index, get_class_name()};
    }
public:
    AutoIndentEnd(Structure &structure, std::unique_ptr<ParserNode> &&only_child)
        : AutoIndent(structure, std::move(only_child)) {
    }
};


class OptionalNode : public ZeroOrMore {
    PARSER_NODE_CLASS(OptionalNode);
public:
    OptionalNode(Structure &structure, std::unique_ptr<ParserNode> &&only_child)
        : ZeroOrMore(structure, std::move(only_child), true) {
    }

    bool is_optional_root() const override {
        return true;
    }
};


inline void make_node_list_recurse(NodeList &nl) {
}

template<typename T, typename... Types>
inline void make_node_list_recurse(NodeList &nl, T &&var1, Types &&... var2) {
    nl.emplace_back(std::forward<T&&>(var1));
    make_node_list_recurse(nl, std::forward<Types&&>(var2)...);
}

template<typename T, typename... Types>
inline NodeList make_node_list(T &&var1, Types &&... var2) {
    NodeList nl;
    nl.emplace_back(std::forward<T&&>(var1));
    make_node_list_recurse(nl, std::forward<Types&&>(var2)...);
    return nl;
}


class OneOrMore : public AnyOf {
    PARSER_NODE_CLASS(OneOrMore);

public:
    OneOrMore(Structure &structure, std::unique_ptr<ParserNode> &&child)
        : AnyOf(structure, make_node_list(std::make_unique<ZeroOrMore>(structure, std::move(child)))) {
    }

    int virtual_which_match(char32_t chr) override {
        for (unsigned int i = 0; i < children.size(); i++) {
            if (children[i]->which_match(chr) == 1) {
                return i + 1;
            }
        }
        return 0;
    }

    bool is_optional_root() const override {
        return false;
    }
};


class GreedyString : public Sequence {
    PARSER_NODE_CLASS(GreedyString);
public:
    GreedyString(Structure &structure, const std::u32string &token_str, bool is_hidden = false)
        : Sequence(structure, NodeList()) {
        for (char32_t ch : token_str) {
            if (is_hidden && children.empty()) {
                children.emplace_back(std::make_unique<HiddenConstantCharNode>(structure, token_str.front()));
            } else if (is_hidden) {
                children.emplace_back(std::make_unique<HiddenGreedyConstantCharNode>(structure, ch));
            } else if (children.empty()) {
                children.emplace_back(std::make_unique<ConstantCharNode>(structure, token_str.front()));
            } else {
                children.emplace_back(std::make_unique<GreedyConstantCharNode>(structure, ch));
            }
        }
    }
};

class HiddenGreedyString : public GreedyString {
    PARSER_NODE_CLASS(HiddenGreedyString);

public:
    HiddenGreedyString(Structure &structure, const std::u32string &token_str)
        : GreedyString(structure, token_str, true) {
        hidden_output = true;
    }
};


//PathType = Union[Tuple[()], Tuple[Union[int, str], ...]]

// def limit_str_length(x: str, n: int) -> str:
//     if len(x) > n:
//         x = x[:max(n - 23, 3)] + "..." + x[-20:]
//     return x


/*
def cleanup_string_tokens(token_strings: list[str]) -> list[str]:
    token_strings = token_strings[:]
    for i in range(len(token_strings)):
        s: str = token_strings[i].replace("\r", "\n")
        if s == '<s>' or s == '</s>':
            s = ''
        if s.startswith("<0x") and s.endswith('>'):
            s = chr(int(s[1:-1], 16))
        token_strings[i] = s.replace('\u2581', ' ')
    return token_strings
*/


class NodeStructureGraph {
public:
    mutable InternTable intern_table;
    std::unordered_map<InternedString, std::unique_ptr<Structure>, InternedStringHash> structures;
    InternedString root_structure;

    NodeStructureGraph() = default;
    virtual ~NodeStructureGraph() = default;
    NodeStructureGraph(const NodeStructureGraph&) = delete;
    NodeStructureGraph &operator=(const NodeStructureGraph&) = delete;

    InternedString intern_string(const std::u32string &str) const {
        return intern_table.intern(str);
    }

    const std::u32string &get_string(InternedString str) const {
        return str.get(intern_table);
    }

    Structure &get_root_structure() const {
        return *(structures.find(root_structure)->second);
    }

    Structure &add_structure(const std::u32string &name) {
        InternedString str = intern_string(name);
        this->structures[str] = std::make_unique<Structure>(*this, str);
        return *this->structures[str];
    }

    void init_graph() {
        for (std::unordered_map<InternedString, std::unique_ptr<Structure>>::value_type &kv : structures) {
            Structure &structure = *kv.second;
            structure.construct_indices();
            for (size_t idx = 0; idx < structure.indices.size(); idx++) {
                ParserNode& node = structure.indices[idx];
//                structure.node_to_index[node] = idx;
                node.node_key = node.generate_node_key();
            }
        }
    }

    std::vector<std::reference_wrapper<ParserNode>> get_all_basic_entry_points() const;

    // void calculate_structure_graph(const TokenizerData &tokenizer_data, PrecalculatedRawGraph &r_precalculated_raw_graph);
    //def calculate_structure_graph(self, token_strings: list[str]) -> PrecalculatedRawGraph:
};

inline InternedString Structure::intern_string(const std::u32string &str) const {
    return graph.intern_string(str);
}

inline const std::u32string &Structure::get_string(InternedString str) const {
    return str.get(graph.intern_table);
}

inline Structure::Structure(NodeStructureGraph &graph, InternedString name)
    : StructureNode(*this, *this),
      graph(graph),
      name(name) {
    root_string = intern_string(U"Root");
    end_string = intern_string(U"End");
    end_structure = std::make_unique<EndNode>(*this);
    end_structure->reachable = true;
    end_structure->index = 0;
}


class StructureOpGraph {
    const PrecalculatedRawGraph &precalculated_raw_graph;
    const TokenizerData& tokenizer_data;
    const std::vector<std::u32string> &token_strings;

    std::unordered_map<NodeKeyTuple, std::shared_ptr<OpTupleToToken>, CustomHasher> current_ops;
    std::unordered_map<NodeKeyTuple, std::shared_ptr<NodeKeyToToken>, CustomHasher> outgoing_no_op_states;

public:
    int history_length = 1;

    StructureOpGraph(const PrecalculatedRawGraph &precalculated_raw_graph, bool structure_only, const TokenizerData &tokenizer_data);

private:
    void _explore_neighbors(const NodeKeyTuple &node_key, OpTupleToToken &neighbor_op_set, std::unordered_set<NodeKeyTuple, CustomHasher> &visited, TokenTensor parent_tok_ids=TokenTensor()) const;

    // def get_neighbors_str(const NodeKeyTuple &node_key) -> dict[OpTuple, set[str]]:
    //     neighbor_op_sets: dict[OpTuple, TokenSet] = this->get_neighbors_tok(node_key)
    //     result_str_sets: dict[OpTuple, set[str]] = {}
    //     for op, token_set in neighbor_op_sets.items():
    //         result_str_sets[op] = token_set.to_str_set()
    //     return result_str_sets

public:
    OpTupleToToken get_neighbors_tok(const NodeKeyTuple &node_key) const {
        std::unordered_set<NodeKeyTuple, CustomHasher> visited;
        OpTupleToToken neighbor_op_sets;
        this->_explore_neighbors(node_key, neighbor_op_sets, visited);
        return neighbor_op_sets;
    }
};


struct PrecalculatedStructureGraph {
    std::unique_ptr<TokenizerData> tokenizer_data;
    std::unique_ptr<NodeStructureGraph> structure_graph;
    std::unique_ptr<InternTable> intern_table_holder;

    InternTable &intern_table;
    const std::vector<std::u32string> &token_strings;
    const std::unordered_multimap<std::u32string, unsigned> &tokens_to_id;

//    std::set<InternedString> string_structures;
//    std::set<InternedString> recursive_structures;
    SingleNodeKey root_node_key;
    TokenTensor null_tensor;

    std::unique_ptr<PrecalculatedRawGraph> precalculated_raw_graph;
//    std::unique_ptr<StructureOpGraph> op_struct_tok;
//    StructureOpGraph op_graph_tok;
    unsigned history_length = 100;
    unsigned max_spaces = 1;

    InternedString indentStr = intern_table.intern(U"AutoIndent");
    InternedString indentEndStr = intern_table.intern(U"AutoIndentEnd");
    InternedString endNodeStr = intern_table.intern(U"EndNode");
    InternedString spaceStr = intern_table.intern(U" ");

    std::vector<NodeKeyTuple> precalculated_state_keys;
    std::unordered_map<NodeKeyTuple, unsigned, CustomHasher> precalculated_indices_by_state_key;
    std::unique_ptr<uint8_t[]> precalculated_vectors_tensor;

    typedef std::pair<NodeKeyTuple, unsigned> PerTokenKey;
    typedef std::pair<OpTuple, SingleNodeKey> PerTokenOutput;
    typedef std::pair<PerTokenKey, PerTokenOutput> PerTokenDataPair;
    typedef std::unordered_map<PerTokenKey, PerTokenOutput, CustomHasher> GraphType;
    GraphType token_output_graph;

    PrecalculatedStructureGraph (std::unique_ptr<TokenizerData> tokenizer_data_, std::unique_ptr<NodeStructureGraph> node_structure_graph_);
    PrecalculatedStructureGraph (std::unique_ptr<TokenizerData> tokenizer_data_, std::unique_ptr<InternTable> intern_table_,
                                 //const std::set<InternedString> &string_structures, const std::set<InternedString> &recursive_structures,
                                 SingleNodeKey root_node_key, unsigned history_length,
                                 std::vector<NodeKeyTuple> &&precalculated_state_keys_,
                                 std::unique_ptr<uint8_t[]> &&precalculated_vectors_tensor_,
                                 const std::vector<PerTokenDataPair> &token_output_graph_);
};

struct StructureExecutionEngine {
    std::shared_ptr<PrecalculatedStructureGraph> precalculated_structure_graph;
    bool use_schema = false;
    int indents_needed = 0;
    int indent_space_size;
    std::vector<SingleNodeKey> struct_stack;
    const PrecalculatedStructureGraph::GraphType &token_output_graph;

    const TokenizerData &tokenizer_data;
    const std::vector<std::u32string> &token_strings;
    const std::unordered_multimap<std::u32string, unsigned> &tokens_to_id;

//    const std::unordered_set<InternedString, InternedStringHash> &string_structures;
//    const std::unordered_set<InternedString, InternedStringHash> &recursive_structures;
    const SingleNodeKey &root_node_key;
//    const StructureOpGraph &op_struct_tok;
//    const StructureOpGraph &op_graph_tok;
    unsigned history_length;
    unsigned max_spaces;

    const std::unordered_map<NodeKeyTuple, unsigned, CustomHasher> &precalculated_indices_by_state_key;
    InternedString indentStr = precalculated_structure_graph->indentStr;
    InternedString indentEndStr = precalculated_structure_graph->indentEndStr;
    InternedString endNodeStr = precalculated_structure_graph->endNodeStr;
    InternedString spaceStr = precalculated_structure_graph->spaceStr;
    const NodeKeyTuple end_key = {{spaceStr, 0, InternedString()}};

    StructureExecutionEngine(const std::shared_ptr<PrecalculatedStructureGraph> &precalculated_structure_graph, int indent_space_size)
        : precalculated_structure_graph(precalculated_structure_graph),
          indent_space_size(indent_space_size),
          token_output_graph(precalculated_structure_graph->token_output_graph),
          tokenizer_data(*precalculated_structure_graph->tokenizer_data),
          token_strings(precalculated_structure_graph->token_strings),
          tokens_to_id(precalculated_structure_graph->tokens_to_id),
//          string_structures(precalculated_structure_graph->string_structures),
//          recursive_structures(precalculated_structure_graph->recursive_structures),
          root_node_key(precalculated_structure_graph->root_node_key),
//          op_struct_tok(precalculated_structure_graph->op_struct_tok),
//          op_graph_tok(precalculated_structure_graph->op_graph_tok),
          history_length(precalculated_structure_graph->history_length),
          max_spaces(precalculated_structure_graph->max_spaces),
          precalculated_indices_by_state_key(precalculated_structure_graph->precalculated_indices_by_state_key) {

        this->init();
    }

    /*
    Example of how get_neighbors works:

    >>> op_struct.get_neighbors((('json', 1),))
    {(('json', 3, 'array'),): {'[', '["', '[]'}, (('json', 2, 'object'),): {'{', '{}', '{"'},
     (('json', 3, 'array'), ('array', 14, 'array')): {'[['},
     (('json', 3, 'array'), ('array', 12, 'number')): {'[-'},
     (('json', 3, 'array'), ('array', 13, 'object')): {'[{'}}
    >>> op_struct.get_neighbors((('json', 0,'end'),))
    {}
    >>> op_struct.get_neighbors((('json', 2,'object'),))
    {(('json', 2, 'object'),): {'{', '{}', '{"'}}
    >>> op_struct.get_neighbors((('json', 3,'array'),))
    {(('json', 3, 'array'),): {'[', '["', '[]'}, (('json', 3, 'array'), ('array', 14, 'array')): {'[['},
     (('json', 3, 'array'), ('array', 12, 'number')): {'[-'},
     (('json', 3, 'array'), ('array', 13, 'object')): {'[{'}}
    */

    void set_state(SingleNodeKey nodes) {
        this->struct_stack.clear();
        struct_stack.emplace_back(nodes);
    }

    void set_state(InternedString root_structure) {
        set_state(SingleNodeKey{root_structure, 1, InternedString()});
    }

    bool set_state(const std::string &root_structure_name) {
        InternedString root_str = precalculated_structure_graph->intern_table.lookup(to_ascii_u32string(root_structure_name));
        if (root_str == InternedString()) {
            return false;
        }
        set_state(SingleNodeKey{root_str, 1, InternedString()});
        return true;
    }

    void init() {
        set_state(this->root_node_key);
    }

    /*int count_recursive_op_nesting(const OpTuple &op_sequence) {
        int count_push = 0;
        if (struct_stack.size() < 2 || struct_stack[struct_stack.size() - 2].nodeType == InternedString()) {
            fprintf(stderr, "Assertion fail: Cannot pop struct when second-last struct key does not have a type.\n");
            return -1;
        }
        bool is_recursive = recursive_structures.count(struct_stack[struct_stack.size() - 2].nodeType) != 0;
        for (StructOperation op : op_sequence) {
            if (op.type == StructOperation::PUSH) {
                if (is_recursive) {
                    count_push += 1;
                } else {
                    is_recursive = false;
                }
            } else if (op.type == StructOperation::POP) {
                if (is_recursive) {
                    count_push -= 1;
                } else {
                    is_recursive = true;
                }
            }
        }
        return count_push;
    }*/

    std::vector<std::u32string> get_logit_weights_str() {
        unsigned tensor_index = get_logit_weights_index();;
        std::vector<std::u32string> possible_token_chars;
        /*
        TokenTensor ret_tensor = precalculated_tokens[tensor_index];
        CPUBitTensor & tensor_data = *ret_tensor;
        for (unsigned token = 0; token < tokenizer_data.token_strings.size(); token++) {
            if (tensor_data.has(token)) {
                possible_token_chars.emplace_back(this->token_strings[token]);
            }
        }
        */
        const std::unique_ptr<uint8_t[]> &precalculated_vectors_tensor = precalculated_structure_graph->precalculated_vectors_tensor;
        size_t token_size = tokenizer_data.token_strings.size();
        for (size_t token = 0; token < token_size; token++) {
            if (precalculated_vectors_tensor[tokenizer_data.padded_token_size * tensor_index + token]) {
                possible_token_chars.emplace_back(this->token_strings[token]);
            }
        }
        return possible_token_chars;
    }

    unsigned get_logit_weights_index()
    {
        std::unordered_map<NodeKeyTuple, unsigned, CustomHasher>::const_iterator iter;
        if (reached_end())
        {
            iter = precalculated_indices_by_state_key.find(end_key);
            if (iter != precalculated_indices_by_state_key.end())
            {
                // isn't this always 0?
                return iter->second;
            }
            return 0;
        }
        NodeKeyTuple node_key(this->struct_stack.size() > (unsigned)history_length ?
            this->struct_stack.end() - history_length :
            this->struct_stack.begin(), this->struct_stack.end());
        unsigned acceptable_token_index = 0;
        /*if (use_schema) {
            acceptable_token_set = TokenTensor(this->tokenizer_data);
            for (const OpTupleToToken::value_type &op_sequence_token_set : op_struct_tok.get_neighbors_tok(node_key)) {
                if (this->struct_stack.size() > 3) {
                    if (this->count_recursive_op_nesting(op_sequence_token_set.first) > 0) {
                        continue;
                    }
                }
                // print(op_sequence, len(token_set));
                acceptable_token_set->update(*op_sequence_token_set.second);
            }
        } else*/ {
            NodeKeyTuple sub_key = node_key;
            while (sub_key.size() > 1 && precalculated_indices_by_state_key.find(sub_key) == precalculated_indices_by_state_key.end()) {
                sub_key.erase(sub_key.begin());
            }
            iter = precalculated_indices_by_state_key.find(sub_key);
            if (iter != precalculated_indices_by_state_key.end()) {
                acceptable_token_index = iter->second;
            } else {
                iter = precalculated_indices_by_state_key.find(end_key);
                if (iter != precalculated_indices_by_state_key.end()) {
                    acceptable_token_index = iter->second;
                }
                // this->reached_end = true;
            }
        }
        // We already exclude space_tokens from indents in precalculated_indices_by_state_key
        if (this->indents_needed > 0 && (this->struct_stack.back().nodeType == indentStr || this->struct_stack.back().nodeType == indentEndStr)) {
            SingleNodeKey space_key = {spaceStr, this->indents_needed, InternedString()};
            if ((unsigned)this->indents_needed > this->max_spaces) {
                space_key.pos = this->max_spaces;
            }
            std::unordered_map<NodeKeyTuple, unsigned, CustomHasher>::const_iterator iter = precalculated_indices_by_state_key.find(NodeKeyTuple{space_key});
            if (iter != precalculated_indices_by_state_key.end()) {
                acceptable_token_index = iter->second;
            }
        }
        return acceptable_token_index;
    }

    const std::pair<OpTuple, SingleNodeKey> *get_precalculated_transition(unsigned selected_token) {
        NodeKeyTuple node_key (this->struct_stack.size() > (unsigned)history_length ?
                               this->struct_stack.end() - history_length :
                               this->struct_stack.begin(), this->struct_stack.end());
        for (unsigned i = 0; i < node_key.size(); i++) {
            std::pair<NodeKeyTuple, unsigned> tmp_key (NodeKeyTuple(node_key.end() -i - 1, node_key.end()), selected_token);
            PrecalculatedStructureGraph::GraphType::const_iterator iter = token_output_graph.find(tmp_key);
            if (iter != token_output_graph.end()) {
                return &iter->second;
            }
        }
        return nullptr;
    }

    SingleNodeKey get_state() const {
        return struct_stack.back();
    }

    NodeKeyTuple get_full_state() const {
        return struct_stack;
    }

    bool reached_end() const {
        return struct_stack.size() == 1 && struct_stack.back().nodeType == endNodeStr;
    }

    SingleNodeKey execute_str(const std::u32string &selected_token_str) {
        TokenizerData::TokensToIdMap::const_iterator tok_iter = tokenizer_data.tokens_to_id.find(selected_token_str);
        if (tok_iter == tokenizer_data.tokens_to_id.end()) {
            return SingleNodeKey{};
        }
        return execute_tok(tok_iter->second);
    }
    SingleNodeKey execute_tok(unsigned selected_token) {
        const std::pair<OpTuple, SingleNodeKey> *transition_data = this->get_precalculated_transition(selected_token);
        if (transition_data == nullptr) {
            // Unable to access the selected token from the current state. We have decoded an invalid state
            // raise Exception("Failed to lookup token " + str(selected_token_str) + " from state " + str(this->struct_stack[-this->history_length:]));
            // fprintf(stderr, "Failed to lookup token!\n");
            return {};
        }
        if (this->struct_stack.empty()) {
            fprintf(stderr, "Empty stack!\n");
            return {};
        }
        OpTuple ops = transition_data->first;
        SingleNodeKey new_state = transition_data->second;

        SingleNodeKey prev_state = this->struct_stack.back();
        this->struct_stack.pop_back();
        for (StructOperation op : ops) {
            if (op.type == StructOperation::POP) {
                if (this->struct_stack.empty()) {
                    fprintf(stderr, "Pop from empty stack!\n");
                    return {};
                }
                this->struct_stack.pop_back();
            } else if (op.type == StructOperation::PUSH) {
                this->struct_stack.emplace_back(op.pushStruct);
            }
        }
        this->struct_stack.emplace_back(new_state);
        if (new_state.nodeType == indentStr || new_state.nodeType == indentEndStr) {
            const std::u32string &selected_token_str = tokenizer_data.token_strings[selected_token];
            if (prev_state != new_state) {
                // Subtract 1 to account for combined space tokens.
                this->indents_needed = this->indent_space_size * (this->struct_stack.size() - (new_state.nodeType == indentEndStr ? 2 : 1)) - 1;
                std::u32string::size_type spacestart = selected_token_str.find_first_of(U' ');
                if (spacestart != std::u32string::npos) {
                    this->indents_needed -= (int)(selected_token_str.size() - spacestart);
                }
            } else if (selected_token_str.find_first_not_of(U' ') == std::u32string::npos) {
                this->indents_needed -= (int)selected_token_str.size();
            } else {
                //raise Exception("Invalid indentation " + str(selected_token_str) + " at state " + str(this->struct_stack) + ": " + str(this->indents_needed))
                // FIXME: Occurs in Llama 3. Likely a case where a token has combined newline and space.
                this->indents_needed -= 1;
            }
        }
        return new_state;
    }
};

namespace StructureDebug {
    struct char_to_char32 : public std::codecvt<char32_t, char, std::mbstate_t> {
        using std::codecvt<char32_t, char, std::mbstate_t>::codecvt;
        ~char_to_char32() = default;
    };

    inline std::ostream &operator<<(std::ostream &os, const std::u32string &str_vec) {
        std::wstring_convert<char_to_char32, char32_t> utf32_to_utf8_converter;
        return os << utf32_to_utf8_converter.to_bytes(str_vec);
    }

    inline std::ostream &operator<<(std::ostream &os, const char32_t *str_vec) {
        std::wstring_convert<char_to_char32, char32_t> utf32_to_utf8_converter;
        return os << utf32_to_utf8_converter.to_bytes(str_vec);
    }

    inline std::ostream &operator<<(std::ostream &os, char32_t ch) {
        std::wstring_convert<char_to_char32, char32_t> utf32_to_utf8_converter;
        return os << utf32_to_utf8_converter.to_bytes(ch);
    }

    inline std::ostream &operator<<(std::ostream &os, const TokenTensor &tt) {
        os << '{';
        bool first = true;
        unsigned tt_size = tt->bit_vector.size() << 9;
        for (unsigned token = 0; token < tt_size; token++) {
            if (tt->has(token)) {
                if (!first) {
                    os << ',';
                }
                first = false;
                os << token;
            }
        }
        os << '}';
        return os;
    }

    inline thread_local int count_limit = 100;
    inline std::ostream &operator<<(std::ostream &os, const std::vector <std::u32string> &str_vec) {
        os << '{';
        bool first = true;
        int i = 0;
        for (const std::u32string &str: str_vec) {
            if (!first) {
                os << ',';
            }
            first = false;
            os << '"' << str << '"';
            i++;
            if (count_limit >= 0 and i >= count_limit) {
                os << " ..." << ((int)str_vec.size() - count_limit);
                break;
            }
        }
        os << '}';
        return os;
    }

#ifdef DEBUG_INTERNED_STRINGS
    inline const thread_local InternTable *intern_table;
#endif

    inline std::ostream &operator<<(std::ostream &os, const InternedString &str) {
#ifdef DEBUG_INTERNED_STRINGS
        if (intern_table != nullptr) {
            os << intern_table->get(str);
        }
        else
#endif
        {
            os << "{INTERN:" << str.stringId << '}';
        }
        return os;
    }

    inline std::ostream &operator<<(std::ostream &os, const SingleNodeKey &single_key) {
        os << "&[" << single_key.structName << ":" << single_key.pos << " " << single_key.nodeType << "]";
        return os;
    }

    inline std::ostream &operator<<(std::ostream &os, const NodeKeyTuple &key_vec) {
        os << "(";
        bool first = true;
        for (const SingleNodeKey &key: key_vec) {
            if (!first) {
                os << ',';
            }
            first = false;
            os << key;
        }
        os << ")";
        return os;
    }

    inline std::ostream &operator<<(std::ostream &os, const ParserNode &node) {
        const SingleNodeKey &key = node.node_key;
        os << node.get_class_name() << key << "@" << node.path;
        return os;
    }

    inline std::ostream &operator<<(std::ostream &os, const ParserNode *node) {
        if (!node) {
            os << "(null)";
        } else {
            const SingleNodeKey &key = node->node_key;
            os << node->get_class_name() << key << "@" << node->path;
        }
        return os;
    }

    inline std::ostream &operator<<(std::ostream &os, const StructOperation &op) {
        switch (op.type) {
            case StructOperation::APPEND_CHAR: {
                char32_t data[2] = {op.appendChar, 0};
                os << data;
                break;
            }
            case StructOperation::PUSH:
                os << "PUSH " << op.pushStruct;
                break;
            case StructOperation::POP:
                os << "POP";
                break;
            default:
                fprintf(stderr, "Invalid operation!\n");
        }
        return os;
    }

    inline std::ostream &operator<<(std::ostream &os, const OpTuple &op_vec) {
        os << "%[";
        bool first = true;
        for (const StructOperation &op: op_vec) {
            if (!first) {
                os << ',';
            }
            first = false;
            os << op;
        }
        os << "]";
        return os;
    }
}

template <typename T1, typename T2, typename ...Args>
struct PrintArgs<T1, T2, Args...> {
    void operator()(const T1 &t1, const T2 &t2, Args... args) {
        std::cout << t1 << ",";
        PrintArgs<T2, Args...>()(t2, args...);
    }
};

template <typename T1>
struct PrintArgs<T1> {
    void operator()(const T1 &t1) {
        std::cout << t1;
    }
};

template <>
struct PrintArgs<> {
    void operator()() {
    }
};

}

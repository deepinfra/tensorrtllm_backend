/* structure_execution_engine_cli.cc
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
#include "structured_execution_engine_cli.h"
#include "structured_execution_safetensors.h"

namespace structured_execution {

using namespace StructureDebug;

static bool verbose_test = true;

void exec(StructureExecutionEngine &engine, std::u32string token) {
    std::stringstream ss;
    std::ostream &os = verbose_test ? std::cout : ss;
    if (token.empty()) {
        os << engine.get_full_state() << std::endl;
        os << engine.get_logit_weights_index() << std::endl;
        os << engine.get_logit_weights_str() << std::endl;
    }
    while (!token.empty()) {
        std::u32string sub_token = token;
        while (!sub_token.empty()) {
            os << "exec(\"" << sub_token << "\"): " << std::endl;
            SingleNodeKey new_key = engine.execute_str(sub_token);
            os << new_key << std::endl;
            if (new_key.structName != InternedString()) {
                break;
            }
            sub_token.pop_back();
        }
        if (engine.reached_end()) {
            std::cout << " *** Completed Test" << std::endl;
        }
        if (verbose_test) {
            os << engine.get_full_state() << std::endl;
            os << engine.get_logit_weights_index() << std::endl;
            os << engine.get_logit_weights_str() << std::endl;
        }
        if (sub_token.empty()) {
            break;
        }
        token.erase(token.begin(), token.begin() + sub_token.size());
    }
}

void execChained(StructureExecutionEngine &engine, int cnt) {
    std::stringstream ss;
    std::ostream &os = verbose_test ? std::cout : ss;
    if (engine.reached_end()) {
        std::cout << " *** REACHED END ***" << std::endl;
    }
    os << engine.get_full_state() << std::endl;
    os << engine.get_logit_weights_index() << std::endl;
    os << engine.get_logit_weights_str() << std::endl;
    int whichTokenMask = (int)engine.get_logit_weights_index();
    int selectedToken = 0;
    if (whichTokenMask >= 0) {
        uint8_t *mask = engine.precalculated_structure_graph->precalculated_vectors_tensor.get() + whichTokenMask * engine.tokenizer_data.padded_token_size;
        for (int i = 0; i < engine.tokenizer_data.padded_token_size; i++) {
            if (mask[i]) {
                selectedToken = i;
                if (cnt-- == 0) {
                    break;
                }
            }
        }
        os << "Selected token " << selectedToken << " \"" << engine.token_strings[selectedToken] << "\"" << std::endl;
        os << "exec(\"" << engine.token_strings[selectedToken] << "\"): " << engine.execute_tok(selectedToken) << std::endl;
    }
}

std::vector<std::u32string> read_strings(const char *bin_filename) {
    FILE *fp = fopen(bin_filename, "rb");
    if (fp == nullptr) {
        fprintf(stderr, "Failed to open input file %s: %d\n", bin_filename, errno);
        return {};
    }
    fseek(fp, 0, SEEK_END);
    uint64_t filepos = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint64_t num_chars = ((filepos + sizeof(char32_t) - 1) / sizeof(char32_t)) + 1;
    char32_t * buf;
    buf = (char32_t*)calloc(sizeof(char32_t), num_chars + 1);
    std::vector<std::u32string> token_strings;
    if (fread(buf, 1, filepos, fp) == filepos) {
        const char32_t *p = buf, *end = buf + num_chars;
        do {
            token_strings.emplace_back(p);
            p += token_strings.back().size() + 1;
        } while (p < end);
    } else {
        fprintf(stderr, "Failed to read whole input file %s: %d\n", bin_filename, errno);
    }
    fclose(fp);
    free(buf);
    return token_strings;
}
}

using namespace structured_execution;

int main(int argc, char**argv) {
    std::vector<std::u32string> token_strings = {
        //U"", U"", U"[", U"\n", U"]", U"a", U"[]", U"[\n", U" "
    U"", U"", U"</s>", U"\n", U"{\n", U"[\n", U"!", U"\"", U"{", U"{\"", U"{{", U"{}", U"|", U"}", U"}}", U"~", U"[-",
    U" ", U"  ", U"   ", U"    ", U"", U"\n ", U"\n", U",\n  ", U"\"\"", U"\"\"\"", U"\"\"\":", U"\"\"\",", U"\"\",",
    U"\"\":", U"[],", U"[][]", U"\")", U"\"))", U"\"));", U"\"),", U"\").", U"\");", U"\")]", U"\")`", U"\"false",
    U"\"true", U"123", U"ull", U" {\n", U" {", U" [", U"[\n", U" tru", U" fal", U" nu", U"{\".", U"\"+", U"\",",
    U"\",\"", U":\"\",", U"\":\"\",\"", U":\"\",\"", U"\":\"\",", U"\"\":\"\"", U"\".", U"\".$", U"\"/", U"\":",
    U"\":\"", U"\":{\"", U"\";", U"\"=>", U"\">", U"\"><", U"\"></", U"\":",
    U"\"?", U"\"?>", U"\"\\", U"\"]", U"\"])", U"\"],", U"\"].", U"\"];", U"\"`", U"\"}", U"\"},", U"#", U"$",
    U"%", U"&", U"\'", U"(", U")", U"*", U"+", U",", U", ", U",\n", U",\"", U",-", U",[", U"-", U".", U"/",
    U"0", U"1", U"2", U"3", U"4", U"5", U"6", U"7", U"8", U"9", U":", U";", U"<", U"=", U">", U"?", U"@",
    U"A", U"B", U"C", U"D", U"E", U"F", U"G", U"H", U"I", U"J", U"K", U"L", U"M", U"N", U"O", U"P", U"Q",
    U"R", U"S", U"T", U"U", U"V", U"W", U"X", U"Y", U"Z", U"[", U"[\"", U"[[", U"[]", U"[{", U"\\", U"]", U"],",
    U"],[", U"]]", U"^", U"_", U"`", U"a", U"b", U"c", U"d", U"e", U"f", U"fa", U"ls", U"g", U"h", U"i", U"j",
    U"k", U"l", U"m", U"n", U"nul", U"o", U"p", U"q", U"r", U"s", U"t", U"true", U"u", U"v", U"w", U"x", U"y", U"z"
    };
    std::shared_ptr<PrecalculatedStructureGraph> graph;
    if (argc == 1 || (argc > 1 && strcmp(argv[1], "--help") == 0)) {
        std::cout << "Usage: " << argv[0] << " [--help]" << std::endl
                  << "             Run test and help" << std::endl;
        std::cout << "       " << argv[0] << " filename.safetensors" << std::endl
                  << "             Run test on safetensors model." << std::endl;
        std::cout << "       " << argv[0] << " tokens_file" << std::endl
                  << "             Run test on NUL-separated UTF-32 tokens" << std::endl;
        std::cout << "       " << argv[0] << " tokens_file filename.safetensors [--runtest]" << std::endl
                  << "             Read NUL-separated UTF-32 tokens and generate safetensors" << std::endl;
        if (argc > 1) {
            return 0;
        }
    }
    if (argc > 1 && strstr(argv[1], ".safetensors") != nullptr) {
        FILE *fp = fopen(argv[1], "rb");
        graph = parse_graph(fp);
        if (!graph) {
            fprintf(stderr, "Failed to read input tensors %s: %d\n", argv[1], errno);
            fclose(fp);
            return 1;
        }
        fclose(fp);
    } else {
        if (argc > 1 && *argv[1]) {
            token_strings = read_strings(argv[1]);
        }
        graph = std::make_shared<PrecalculatedStructureGraph>(
                std::make_unique<TokenizerData>(token_strings),
                std::make_unique<JsonNodeStructureGraph>());
    }
    verbose_test = (argc <= 2 || (argc == 4 && strcmp(argv[3], "--runtest") == 0));
    intern_table = &graph->intern_table;
    std::unique_ptr<StructureExecutionEngine> engine = std::make_unique<StructureExecutionEngine>(graph, 2);
//    printf("Hello, World\n");
    exec(*engine, U"");
    /*
    exec(*engine, U"{");
    exec(*engine, U"\"");
    exec(*engine, U"H");
    exec(*engine, U"\":");
    exec(*engine, U"[-");
    exec(*engine, U"3");
    exec(*engine, U"}");
     */
    exec(*engine, U" {");
    exec(*engine, U"\"");
    exec(*engine, U"[\"");
    exec(*engine, U":");
    exec(*engine, U"nul");
    exec(*engine, U"l");
    exec(*engine, U",\n");
    exec(*engine, U"  ");
    exec(*engine, U"  ");
    exec(*engine, U"\"");
    exec(*engine, U"H");
    exec(*engine, U"\"");
    exec(*engine, U":");
    exec(*engine, U"3");
    exec(*engine, U"}");
    exec(*engine, U"\n");
    engine->init();
    engine->set_state("markdown_json");
    for (int i = 0; i < 15; ++i) {
        execChained(*engine, 2 * i);
    }
    if (argc > 2 && strstr(argv[2], ".safetensors") != nullptr) {
        FILE *fp = fopen(argv[2], "wb");
        if (!write_graph(fp, *graph)) {
            fprintf(stderr, "Failed to write output tensors %s: %d\n", argv[2], errno);
            fclose(fp);
                return 2;
        }
        fflush(fp);
        fclose(fp);
        std::cout << " *** Wrote safetensors to " << argv[2] << std::endl;
    }
    return 0;
}

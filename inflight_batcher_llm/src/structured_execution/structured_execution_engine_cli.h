/* structure_execution_engine_cli.h
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

#include "structured_execution_engine.h"

namespace structured_execution
{

class JsonNodeStructureGraph : public NodeStructureGraph {
public:
    Structure& number;
    Structure& string;
    Structure& array;
    Structure& object;
    Structure& constant;
    Structure& json;
    Structure& json_string;
    Structure& json_number;
    Structure& json_constant;
    Structure& json_object;
    Structure& json_array;
    Structure& json_any;
    Structure& markdown_json;
    Structure& markdown_json_object;
    Structure& markdown_json_array;
    Structure& function_call;

    JsonNodeStructureGraph()
        : number(add_structure(U"number")),
        string(add_structure(U"string")),
        array(add_structure(U"array")),
        object(add_structure(U"object")),
        constant(add_structure(U"constant")),
        json(add_structure(U"json")),
        json_string(add_structure(U"json_string")),
        json_number(add_structure(U"json_number")),
        json_constant(add_structure(U"json_constant")),
        json_object(add_structure(U"json_object")),
        json_array(add_structure(U"json_array")),
        json_any(add_structure(U"json_any")),
        markdown_json(add_structure(U"markdown_json")),
        markdown_json_object(add_structure(U"markdown_json_object")),
        markdown_json_array(add_structure(U"markdown_json_array")),
        function_call(add_structure(U"function_call"))
    {
        StructureDebug::intern_table = &intern_table;

        auto value = [&](Structure &s) {
            return std::make_unique<AnyOf>(s, make_node_list(
                    std::make_unique<StructureNode>(s, this->string),
                    std::make_unique<StructureNode>(s, this->constant),
                    std::make_unique<StructureNode>(s, this->object),
                    std::make_unique<StructureNode>(s, this->array),
                    std::make_unique<StructureNode>(s, this->number)
            ));
        };
        auto onenine = [&](Structure &s) { return std::make_unique<ConstantCharNode>(s, '1', '9'); };
        auto digit = [&](Structure &s) { return std::make_unique<ConstantCharNode>(s, '0', '9'); };
        auto fraction = [&](Structure &s) {
            return std::make_unique<Sequence>(s, make_node_list(
                    std::make_unique<ConstantCharNode>(s, '.'),
                    digit(s),
                    std::make_unique<ZeroOrMore>(s, digit(s))
            ));
        };
        auto integer = [&](Structure &s) {
            return std::make_unique<Sequence>(s, make_node_list(
                    std::make_unique<OptionalNode>(s,
                                                   std::make_unique<ConstantCharNode>(s, '-')),
                    std::make_unique<AnyOf>(s, make_node_list(
                            std::make_unique<ConstantCharNode>(s, '0'),
                            std::make_unique<Sequence>(s, make_node_list(
                                    onenine(s),
                                    std::make_unique<ZeroOrMore>(s, digit(s))
                            ))
                    ))
            ));
        };
        auto exponent_sign = [&](Structure &s) { return std::make_unique<AnyOfCharNode>(s, U"+-"); };
        auto exponent = [&](Structure &s) {
            return std::make_unique<Sequence>(s, make_node_list(
                    std::make_unique<AnyOfCharNode>(s, U"eE"),
                    std::make_unique<AnyOf>(s, make_node_list(
                            std::make_unique<Sequence>(s, make_node_list(
                                    exponent_sign(s),
                                    digit(s)
                            )),
                            digit(s)
                    )),
                    std::make_unique<ZeroOrMore>(s, digit(s))
            ));
        };
        // Equivalent to (trange('0', '9'), trange('a', 'f'), trange('A', 'F'))
        auto hex_digit = [&](Structure &s) { return std::make_unique<AnyOfCharNode>(s, U"0123456789abcdefABCDEF"); };
        auto escape = [&](Structure &s) {
            return std::make_unique<AnyOf>(s, make_node_list(
                    std::make_unique<AnyOfCharNode>(s, U"\"\\/bfnrt"),
                    std::make_unique<Sequence>(s, make_node_list(
                            std::make_unique<ConstantCharNode>(s, 'u'),
                            hex_digit(s),
                            hex_digit(s),
                            hex_digit(s),
                            hex_digit(s)
                    ))
            ));
        };
        // std::make_unique<ConstantCharNode>(s, U'[^\u0001-\u001f\"\\]'), ['\\', escape])
        auto character = [&](Structure &s) {
            return std::make_unique<AnyOf>(s, make_node_list(
                    std::make_unique<ConstantCharNode>(s, U' ', U'!'), // 32, 33 (exclude 34 '\"')
                    std::make_unique<ConstantCharNode>(s, U'\"' + 1, U'\\' - 1), // 35..91 (exclude 92 '\\')
                    std::make_unique<Sequence>(s, make_node_list(
                            std::make_unique<ConstantCharNode>(s, U'\\'),
                            escape(s)
                    )),
                    std::make_unique<ConstantCharNode>(s, U'\\' + 1, U'\U0010FFFF') // 93..
            ));
        };

        auto &element = value;
        auto member = [&](Structure &s) {
            return std::make_unique<Sequence>(s, make_node_list(
                    std::make_unique<StructureNode>(s, this->string),
                    std::make_unique<ConstantCharNode>(s, ':'),
                    std::make_unique<OptionalNode>(s, std::make_unique<ConstantCharNode>(s, ' ')),
                    element(s)
            ));
        };

        string.permit_hybrid_ops = false; // Do not allow creating strings with letters in front.
        string.root_node = std::make_unique<Sequence>(string, make_node_list(
                std::make_unique<HiddenConstantCharNode>(string, '"'),
                // TODO: After we add schemas, return this to ZeroOrMore to allow empty strings
                // Models tend to create empty strings when confused. This should prevent that.
                std::make_unique<OneOrMore>(string, character(string)),
                std::make_unique<HiddenConstantCharNode>(string, '"')
        ));

        number.root_node = std::make_unique<Sequence>(number, make_node_list(
                integer(number),
                std::make_unique<OptionalNode>(number, fraction(number)),
                std::make_unique<OptionalNode>(number, exponent(number))
        ));

        constant.root_node = std::make_unique<AnyOf>(constant, make_node_list(
                std::make_unique<GreedyString>(constant, U"true"),
                std::make_unique<GreedyString>(constant, U"false"),
                std::make_unique<GreedyString>(constant, U"null")));

        // Forbid empty objects...
        // this->init_structure(this->array, ['[', (']', elements + [OptionalNode(['\n', AutoIndent(' ')]), ']'])])
        // this->init_structure(this->object, ['{', ('}', members + [OptionalNode(['\n', AutoIndent(' ')]), '}'])])
        {
            Structure &s = array;
            array.root_node = std::make_unique<Sequence>(s, make_node_list(
                    std::make_unique<ConstantCharNode>(s, '['),
                    std::make_unique<OptionalNode>(s, std::make_unique<Sequence>(s, make_node_list(
                            std::make_unique<ConstantCharNode>(s, '\n'),
                            std::make_unique<AutoIndent>(s, std::make_unique<ConstantCharNode>(s, ' '))
                    ))),
                    element(s),
                    std::make_unique<ZeroOrMore>(s, std::make_unique<Sequence>(s, make_node_list(
                            std::make_unique<ConstantCharNode>(s, ','),
                            std::make_unique<OptionalNode>(s, std::make_unique<AnyOf>(s, make_node_list(
                                    std::make_unique<ConstantCharNode>(s, ' '),
                                    std::make_unique<Sequence>(s, make_node_list(
                                            std::make_unique<ConstantCharNode>(s, '\n'),
                                            std::make_unique<AutoIndent>(s, std::make_unique<ConstantCharNode>(s, ' '))
                                    ))
                            ))),
                            element(s)
                    ))),
                    std::make_unique<OptionalNode>(s, std::make_unique<Sequence>(s, make_node_list(
                            std::make_unique<ConstantCharNode>(s, '\n'),
                            std::make_unique<AutoIndentEnd>(s, std::make_unique<ConstantCharNode>(s, ' '))
                    ))),
                    std::make_unique<ConstantCharNode>(s, ']')
            ));
        }

        {
            Structure &s = object;
            object.root_node = std::make_unique<Sequence>(s, make_node_list(
                    std::make_unique<ConstantCharNode>(s, '{'),
                    std::make_unique<OptionalNode>(s, std::make_unique<Sequence>(s, make_node_list(
                            std::make_unique<ConstantCharNode>(s, '\n'),
                            std::make_unique<AutoIndent>(s, std::make_unique<ConstantCharNode>(s, ' '))
                    ))),
                    member(s),
                    std::make_unique<ZeroOrMore>(s, std::make_unique<Sequence>(s, make_node_list(
                            std::make_unique<ConstantCharNode>(s, ','),
                            std::make_unique<OptionalNode>(s, std::make_unique<AnyOf>(s, make_node_list(
                                    std::make_unique<ConstantCharNode>(s, ' '),
                                    std::make_unique<Sequence>(s, make_node_list(
                                            std::make_unique<ConstantCharNode>(s, '\n'),
                                            std::make_unique<AutoIndent>(s, std::make_unique<ConstantCharNode>(s, ' '))
                                    ))
                            ))),
                            member(s)
                    ))),
                    std::make_unique<OptionalNode>(s, std::make_unique<Sequence>(s, make_node_list(
                            std::make_unique<ConstantCharNode>(s, '\n'),
                            std::make_unique<AutoIndentEnd>(s, std::make_unique<ConstantCharNode>(s, ' '))
                    ))),
                    std::make_unique<ConstantCharNode>(s, '}')
            ));
        }

        // to allow either array or object.
        json.root_node = std::make_unique<Sequence>(json, make_node_list(
            std::make_unique<AnyOf>(json, make_node_list(
                std::make_unique<StructureNode>(json, this->object),
                std::make_unique<StructureNode>(json, this->array))),
            std::make_unique<ConstantCharNode>(json, '\n')));
        json_string.root_node = std::make_unique<Sequence>(json_string, make_node_list(
            std::make_unique<StructureNode>(json_string, this->string),
            std::make_unique<ConstantCharNode>(json_string, '\n')));
        json_number.root_node = std::make_unique<Sequence>(json_number, make_node_list(
            std::make_unique<StructureNode>(json_number, this->number),
            std::make_unique<ConstantCharNode>(json_number, '\n')));
        json_constant.root_node = std::make_unique<Sequence>(json_constant, make_node_list(
            std::make_unique<StructureNode>(json_constant, this->constant),
            std::make_unique<ConstantCharNode>(json_constant, '\n')));
        json_object.root_node = std::make_unique<Sequence>(json_object, make_node_list(
            std::make_unique<StructureNode>(json_object, this->object),
            std::make_unique<ConstantCharNode>(json_object, '\n')));
        json_array.root_node = std::make_unique<Sequence>(json_array, make_node_list(
            std::make_unique<StructureNode>(json_array, this->array),
            std::make_unique<ConstantCharNode>(json_array, '\n')));
        json_any.root_node = std::make_unique<Sequence>(json_any, make_node_list(
            std::make_unique<AnyOf>(json_any, make_node_list(
                std::make_unique<StructureNode>(json_any, this->string),
                std::make_unique<StructureNode>(json_any, this->number),
                std::make_unique<StructureNode>(json_any, this->constant),
                std::make_unique<StructureNode>(json_any, this->object),
                std::make_unique<StructureNode>(json_any, this->array))),
            std::make_unique<ConstantCharNode>(json_any, '\n')));

        auto markdown = [&](Structure &s, std::unique_ptr<ParserNode> node) {
            return std::make_unique<Sequence>(s, make_node_list(
                std::make_unique<GreedyString>(s, U"```"),
                std::make_unique<GreedyString>(s, U"json"),
                std::make_unique<ConstantCharNode>(s, '\n'),
                std::forward<std::unique_ptr<ParserNode>>(node),
                std::make_unique<ConstantCharNode>(s, '\n'),
                std::make_unique<GreedyString>(s, U"```"),
                std::make_unique<ConstantCharNode>(s, '\n')));
        };

        markdown_json.root_node = markdown(markdown_json,
            std::make_unique<AnyOf>(markdown_json, make_node_list(
                std::make_unique<StructureNode>(markdown_json, this->object),
                std::make_unique<StructureNode>(markdown_json, this->array))));

        markdown_json_object.root_node = markdown(markdown_json_object,
            std::make_unique<StructureNode>(markdown_json_object, this->object));

        markdown_json_array.root_node = markdown(markdown_json_array,
            std::make_unique<StructureNode>(markdown_json_array, this->array));

        // std::make_unique<ConstantCharNode>(s, U'[^\u0001-\u001f\"\\]'), ['\\', escape])
        auto function_name_character = [&](Structure &s) {
            return std::make_unique<AnyOf>(s, make_node_list(
                    std::make_unique<ConstantCharNode>(s, U'#', U';'), // 32, 33 (exclude 34 '\"','!',' ','<','=','>')
                    std::make_unique<ConstantCharNode>(s, U'?' + 1, U'\U0010FFFF')
            ));
        };

        function_call.root_node = std::make_unique<Sequence>(function_call, make_node_list(
            std::make_unique<HiddenConstantCharNode>(function_call, '<'),
            std::make_unique<HiddenGreedyString>(function_call, U"function"),
            std::make_unique<HiddenConstantCharNode>(function_call, '='),
            function_name_character(function_call),
            std::make_unique<ZeroOrMore>(function_call, function_name_character(function_call)),
            std::make_unique<HiddenConstantCharNode>(function_call, '>'),
            std::make_unique<StructureNode>(function_call, this->object),
            std::make_unique<HiddenConstantCharNode>(function_call, '<'),
            std::make_unique<HiddenGreedyString>(function_call, U"/function"),
            std::make_unique<HiddenConstantCharNode>(function_call, '>')));

        /*
        {
            constant.root_node = std::make_unique<ConstantCharNode>(constant, 'a');
            string.root_node = std::make_unique<ConstantCharNode>(string, 'a');
            number.root_node = std::make_unique<ConstantCharNode>(number, 'a');
            object.root_node = std::make_unique<ConstantCharNode>(object, 'a');
            array.root_node = std::make_unique<ConstantCharNode>(array, 'a');
            Structure &s = object_or_array;
            object_or_array.root_node = std::make_unique<Sequence>(s, make_node_list(
                    std::make_unique<ConstantCharNode>(s, '['),
                    std::make_unique<OptionalNode>(s, std::make_unique<Sequence>(s, make_node_list(
                            std::make_unique<ConstantCharNode>(s, '\n'),
                            std::make_unique<AutoIndent>(s, std::make_unique<ConstantCharNode>(s, ' '))
                    ))),
                    std::make_unique<ConstantCharNode>(s, ']')
            ));
        }
        */

        this->init_graph();
        function_call.is_string = true;
        root_structure = this->json.name;

        // ParserStructureStack structure_stack(object_or_array);
        // structure_stack.clear_ops();
        // ParserNode * brace = object_or_array.tick(&structure_stack, object_or_array.root_node.get(), '[');
        // ParserNode * newl = object_or_array.tick(&structure_stack, brace, '\n');
        // if (newl) std::cout << "Got newl " << *newl << std::endl; else std::cout << "Null newl" << std::endl;
    }
};

}
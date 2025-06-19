"""Tree-sitter based code parsing for Java and Kotlin."""

import logging
from abc import ABC
from enum import Enum

from tree_sitter import Node
from tree_sitter_language_pack import get_language, get_parser

logger = logging.getLogger(__name__)


class LanguageEnum(str, Enum):
    JAVA = "java"
    KOTLIN = "kotlin"
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    RUST = "rust"
    UNKNOWN = "unknown"


LANGUAGE_QUERIES = {
    LanguageEnum.JAVA: {
        "class_query": """
            (class_declaration
                name: (identifier) @class.name)
        """,
        "method_query": """
            [
                (method_declaration
                    name: (identifier) @method.name)
                (constructor_declaration
                    name: (identifier) @method.name)
            ]
        """,
        "doc_query": """
            ((block_comment) @comment)
        """,
    },
    LanguageEnum.KOTLIN: {
        "class_query": """
            [
                (class_declaration
                    (type_identifier) @class.name)
                (object_declaration
                    (type_identifier) @class.name)
            ]
        """,
        "method_query": """
            (function_declaration
                (simple_identifier) @method.name)
        """,
        "doc_query": """
            (multiline_comment) @comment
        """,
    },
    LanguageEnum.PYTHON: {
        "class_query": """
            (class_definition
                name: (identifier) @class.name)
        """,
        "method_query": """
            (function_definition
                name: (identifier) @method.name)
        """,
        "doc_query": """
            (expression_statement (string) @comment)
        """,
    },
    LanguageEnum.TYPESCRIPT: {
        "class_query": """
            [
                (class_declaration
                    name: (type_identifier) @class.name)
                (interface_declaration
                    name: (type_identifier) @class.name)
            ]
        """,
        "method_query": """
            [
                (method_definition
                    name: (property_identifier) @method.name)
                (function_declaration
                    name: (identifier) @method.name)
                (method_signature
                    name: (property_identifier) @method.name)
            ]
        """,
        "doc_query": """
            (comment) @comment
        """,
    },
    LanguageEnum.JAVASCRIPT: {
        "class_query": """
            (class_declaration
                name: (identifier) @class.name)
        """,
        "method_query": """
            [
                (method_definition
                    name: (property_identifier) @method.name)
                (function_declaration
                    name: (identifier) @method.name)
            ]
        """,
        "doc_query": """
            (comment) @comment
        """,
    },
    LanguageEnum.CPP: {
        "class_query": """
            [
                (class_specifier
                    name: (type_identifier) @class.name)
                (struct_specifier
                    name: (type_identifier) @class.name)
            ]
        """,
        "method_query": """
            [
                (function_definition
                    declarator: (function_declarator
                        declarator: (identifier) @method.name))
                (declaration
                    declarator: (function_declarator
                        declarator: (identifier) @method.name))
            ]
        """,
        "doc_query": """
            (comment) @comment
        """,
    },
    LanguageEnum.RUST: {
        "class_query": """
            [
                (struct_item
                    name: (type_identifier) @class.name)
                (enum_item
                    name: (type_identifier) @class.name)
                (trait_item
                    name: (type_identifier) @class.name)
                (impl_item
                    trait: (type_identifier) @class.name)
            ]
        """,
        "method_query": """
            (function_item
                name: (identifier) @method.name)
        """,
        "doc_query": """
            (line_comment) @comment
        """,
    },
}


class TreesitterMethodNode:
    def __init__(
        self,
        name: str,
        doc_comment: str,
        method_source_code: str,
        node,
        class_name: str | None = None,
    ):
        self.name = name
        self.doc_comment = doc_comment
        self.method_source_code = method_source_code
        self.node = node
        self.class_name = class_name


class TreesitterClassNode:
    def __init__(
        self,
        name: str,
        method_declarations: list,
        node,
    ):
        self.name = name
        self.source_code = node.text.decode()
        self.method_declarations = method_declarations
        self.node = node


class Treesitter(ABC):
    def __init__(self, language: LanguageEnum):
        self.language_enum = language

        # Map our language enum to tree-sitter language names
        language_mapping = {
            "java": "java",
            "kotlin": "kotlin",
            "python": "python",
            "typescript": "typescript",
            "javascript": "javascript",
            "cpp": "cpp",
            "rust": "rust",
        }

        language_name = language_mapping.get(language.value.lower())
        if not language_name:
            raise ValueError(f"Unsupported language: {language}")

        try:
            self.parser = get_parser(language_name)  # type: ignore
            self.language_obj = get_language(language_name)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to initialize parser for {language_name}: {e}")
            raise

        self.query_config = LANGUAGE_QUERIES.get(language)
        if not self.query_config:
            raise ValueError(f"Unsupported language: {language}")

        self.class_query = self.language_obj.query(self.query_config["class_query"])
        self.method_query = self.language_obj.query(self.query_config["method_query"])
        self.doc_query = self.language_obj.query(self.query_config["doc_query"])

    @staticmethod
    def create_treesitter(language: LanguageEnum) -> "Treesitter":
        return Treesitter(language)

    def parse(
        self, file_bytes: bytes
    ) -> tuple[list[TreesitterClassNode], list[TreesitterMethodNode]]:
        tree = self.parser.parse(file_bytes)
        root_node = tree.root_node

        class_results = []
        method_results = []

        class_name_by_node = {}
        class_captures = self.class_query.captures(root_node)
        class_nodes = []
        for capture_name, nodes in class_captures.items():
            for node in nodes:
                if capture_name == "class.name":
                    if not node.text:
                        continue
                    class_name = node.text.decode()
                    class_node = node.parent
                    logger.debug(f"Found class: {class_name}")
                    if class_node:
                        class_name_by_node[class_node.id] = class_name
                    method_declarations = self._extract_methods_in_class(class_node)
                    class_results.append(
                        TreesitterClassNode(class_name, method_declarations, class_node)
                    )
                    class_nodes.append(class_node)

        method_captures = self.method_query.captures(root_node)
        for capture_name, nodes in method_captures.items():
            for node in nodes:
                if capture_name in ["method.name", "function.name"]:
                    if not node.text:
                        continue
                    method_name = node.text.decode()
                    method_node = node.parent
                    if not method_node or not method_node.text:
                        continue
                    method_source_code = method_node.text.decode()
                    doc_comment = self._extract_doc_comment(method_node)
                    parent_class_name = None
                    for class_node in class_nodes:
                        if self._is_descendant_of(method_node, class_node):
                            parent_class_name = class_name_by_node[class_node.id]
                            break
                    method_results.append(
                        TreesitterMethodNode(
                            name=method_name,
                            doc_comment=doc_comment,
                            method_source_code=method_source_code,
                            node=method_node,
                            class_name=parent_class_name,
                        )
                    )

        return class_results, method_results

    def _extract_methods_in_class(self, class_node):
        method_declarations = []
        method_captures = self.method_query.captures(class_node)
        for capture_name, nodes in method_captures.items():
            for node in nodes:
                if capture_name in ["method.name", "function.name"]:
                    if not node.parent or not node.parent.text:
                        continue
                    method_declaration = node.parent.text.decode()
                    method_declarations.append(method_declaration)
        return method_declarations

    def _extract_doc_comment(self, node: Node):
        # Search for doc comments preceding the node
        doc_comment = ""
        current_node = node.prev_sibling
        while current_node:
            captures = self.doc_query.captures(current_node)
            if captures:
                for cap_name, cap_nodes in captures.items():
                    for cap_node in cap_nodes:
                        if cap_name == "comment" and cap_node.text:
                            doc_comment = cap_node.text.decode() + "\n" + doc_comment
            elif current_node.type not in [
                "comment",
                "block_comment",
                "line_comment",
                "expression_statement",
            ]:
                break
            current_node = current_node.prev_sibling
        return doc_comment.strip()

    def _is_descendant_of(self, node, ancestor):
        current = node.parent
        while current:
            if current == ancestor:
                return True
            current = current.parent
        return False

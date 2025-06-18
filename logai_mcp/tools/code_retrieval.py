"""Code retrieval tools for finding and extracting method and class implementations."""

import os

import pandas as pd

from logai_mcp.config import get_settings
from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.session import app, logger

from .file_loading import load_files
from .treesitter_parser import LanguageEnum, Treesitter


def _codebase_path_available() -> bool:
    """Check if codebase path is configured and exists."""
    try:
        settings = get_settings()
        if not settings.codebase_path:
            return False
        return os.path.exists(settings.codebase_path)
    except Exception:
        return False


def get_language_enum(language_str: str) -> LanguageEnum | None:
    """Convert language string to LanguageEnum."""
    language_mapping = {
        "java": LanguageEnum.JAVA,
        "kotlin": LanguageEnum.KOTLIN,
        "python": LanguageEnum.PYTHON,
        "typescript": LanguageEnum.TYPESCRIPT,
        "javascript": LanguageEnum.JAVASCRIPT,
        "cpp": LanguageEnum.CPP,
        "rust": LanguageEnum.RUST,
    }
    return language_mapping.get(language_str.lower())


class CodeImplementationResult:
    """Container for method or class implementation results."""

    def __init__(
        self,
        name: str,
        implementation: str,
        file_path: str,
        line_start: int,
        line_end: int,
        doc_comment: str = "",
        class_name: str = "",
        result_type: str = "method",
    ):
        self.name = name
        self.implementation = implementation
        self.file_path = file_path
        self.line_start = line_start
        self.line_end = line_end
        self.doc_comment = doc_comment
        self.class_name = class_name
        self.result_type = result_type  # "method" or "class"

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame compatibility."""
        return {
            "name": self.name,
            "implementation": self.implementation,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "doc_comment": self.doc_comment,
            "class_name": self.class_name,
            "result_type": self.result_type,
        }


class ExactCodeRetriever:
    """Smart code retriever using tree-sitter for exact name matching."""

    def __init__(
        self, codebase_path: str, supported_languages: list[str] | None = None
    ):
        self.codebase_path = codebase_path
        self.supported_languages = supported_languages or ["java", "kotlin"]
        self.file_list = load_files(codebase_path)

    def _is_supported_language(self, language: str) -> bool:
        """Check if the language is in the supported languages list."""
        return language.lower() in [lang.lower() for lang in self.supported_languages]

    def find_method_implementation(
        self, method_name: str, class_name: str | None = None
    ) -> list[CodeImplementationResult]:
        """Find exact method implementation(s) by name.

        Args:
            method_name: The exact name of the method to find
            class_name: Optional class name to narrow down search

        Returns:
            List of CodeImplementationResult objects containing full implementations

        """
        results = []

        for file_path, language in self.file_list:
            # Only process supported languages
            if not self._is_supported_language(language):
                continue

            try:
                with open(file_path, encoding="utf-8") as file:
                    code = file.read()
                    file_bytes = code.encode()

                lang_enum = get_language_enum(language)
                if not lang_enum:
                    logger.warning(f"Unsupported language for parsing: {language}")
                    continue

                treesitter_parser = Treesitter.create_treesitter(lang_enum)
                class_nodes, method_nodes = treesitter_parser.parse(file_bytes)

                for method_node in method_nodes:
                    if method_node.name == method_name:
                        # If class_name is specified, filter by it
                        if class_name and method_node.class_name != class_name:
                            continue

                        # Get line numbers for the method
                        lines = code.split("\n")
                        start_line = method_node.node.start_point[0] + 1
                        end_line = method_node.node.end_point[0] + 1

                        result = CodeImplementationResult(
                            name=method_node.name,
                            implementation=method_node.method_source_code,
                            file_path=file_path,
                            line_start=start_line,
                            line_end=end_line,
                            doc_comment=method_node.doc_comment,
                            class_name=method_node.class_name or "",
                            result_type="method",
                        )
                        results.append(result)

            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue

        return results

    def find_class_implementation(
        self, class_name: str
    ) -> list[CodeImplementationResult]:
        """Find exact class implementation(s) by name.

        Args:
            class_name: The exact name of the class to find

        Returns:
            List of CodeImplementationResult objects containing full implementations

        """
        results = []

        for file_path, language in self.file_list:
            # Only process supported languages
            if not self._is_supported_language(language):
                continue

            try:
                with open(file_path, encoding="utf-8") as file:
                    code = file.read()
                    file_bytes = code.encode()

                lang_enum = get_language_enum(language)
                if not lang_enum:
                    logger.warning(f"Unsupported language for parsing: {language}")
                    continue

                treesitter_parser = Treesitter.create_treesitter(lang_enum)
                class_nodes, method_nodes = treesitter_parser.parse(file_bytes)

                for class_node in class_nodes:
                    if class_node.name == class_name:
                        # Get line numbers for the class
                        start_line = class_node.node.start_point[0] + 1
                        end_line = class_node.node.end_point[0] + 1

                        result = CodeImplementationResult(
                            name=class_node.name,
                            implementation=class_node.source_code,
                            file_path=file_path,
                            line_start=start_line,
                            line_end=end_line,
                            doc_comment="",  # Classes don't have doc comments in our current structure
                            class_name=class_node.name,
                            result_type="class",
                        )
                        results.append(result)

            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue

        return results

    def list_all_methods(self) -> list[tuple[str, str, str]]:
        """List all methods in the codebase.

        Returns:
            List of tuples (method_name, class_name, file_path)

        """
        methods = []

        for file_path, language in self.file_list:
            # Only process supported languages
            if not self._is_supported_language(language):
                continue

            try:
                with open(file_path, encoding="utf-8") as file:
                    code = file.read()
                    file_bytes = code.encode()

                lang_enum = get_language_enum(language)
                if not lang_enum:
                    logger.warning(f"Unsupported language for parsing: {language}")
                    continue

                treesitter_parser = Treesitter.create_treesitter(lang_enum)
                class_nodes, method_nodes = treesitter_parser.parse(file_bytes)

                for method_node in method_nodes:
                    methods.append(
                        (method_node.name, method_node.class_name or "", file_path)
                    )

            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue

        return methods

    def list_all_classes(self) -> list[tuple[str, str]]:
        """List all classes in the codebase.

        Returns:
            List of tuples (class_name, file_path)

        """
        classes = []

        for file_path, language in self.file_list:
            # Only process supported languages
            if not self._is_supported_language(language):
                continue

            try:
                with open(file_path, encoding="utf-8") as file:
                    code = file.read()
                    file_bytes = code.encode()

                lang_enum = get_language_enum(language)
                if not lang_enum:
                    logger.warning(f"Unsupported language for parsing: {language}")
                    continue

                treesitter_parser = Treesitter.create_treesitter(lang_enum)
                class_nodes, method_nodes = treesitter_parser.parse(file_bytes)

                for class_node in class_nodes:
                    classes.append((class_node.name, file_path))

            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue

        return classes


# Implementation functions that will be pushed to the shell


def _find_method_implementation_impl(
    method_name: str,
    class_name: str | None = None,
    codebase_path: str | None = None,
    supported_languages: list[str] | None = None,
) -> pd.DataFrame:
    """Implementation function for finding method implementations.

    Returns:
        pd.DataFrame: DataFrame with method implementation results

    """
    settings = get_settings()

    if not codebase_path:
        codebase_path = settings.codebase_path
    if not supported_languages:
        supported_languages = settings.supported_languages

    if not codebase_path:
        raise ValueError(
            "Codebase path not configured. Set CODEBASE_PATH environment variable."
        )

    if not os.path.exists(codebase_path):
        raise ValueError(f"Codebase path does not exist: {codebase_path}")

    retriever = ExactCodeRetriever(codebase_path, supported_languages)
    results = retriever.find_method_implementation(method_name, class_name)

    if not results:
        return pd.DataFrame(
            columns=[
                "name",
                "implementation",
                "file_path",
                "line_start",
                "line_end",
                "doc_comment",
                "class_name",
                "result_type",
            ]
        )

    # Convert results to DataFrame
    data = [result.to_dict() for result in results]
    df = pd.DataFrame(data)

    logger.info(f"Found {len(results)} method implementations for '{method_name}'")
    return df


def _find_class_implementation_impl(
    class_name: str,
    codebase_path: str | None = None,
    supported_languages: list[str] | None = None,
) -> pd.DataFrame:
    """Implementation function for finding class implementations.

    Returns:
        pd.DataFrame: DataFrame with class implementation results

    """
    settings = get_settings()

    if not codebase_path:
        codebase_path = settings.codebase_path
    if not supported_languages:
        supported_languages = settings.supported_languages

    if not codebase_path:
        raise ValueError(
            "Codebase path not configured. Set CODEBASE_PATH environment variable."
        )

    if not os.path.exists(codebase_path):
        raise ValueError(f"Codebase path does not exist: {codebase_path}")

    retriever = ExactCodeRetriever(codebase_path, supported_languages)
    results = retriever.find_class_implementation(class_name)

    if not results:
        return pd.DataFrame(
            columns=[
                "name",
                "implementation",
                "file_path",
                "line_start",
                "line_end",
                "doc_comment",
                "class_name",
                "result_type",
            ]
        )

    # Convert results to DataFrame
    data = [result.to_dict() for result in results]
    df = pd.DataFrame(data)

    logger.info(f"Found {len(results)} class implementations for '{class_name}'")
    return df


def _list_all_methods_impl(
    codebase_path: str | None = None, supported_languages: list[str] | None = None
) -> pd.DataFrame:
    """Implementation function for listing all methods.

    Returns:
        pd.DataFrame: DataFrame with method information

    """
    settings = get_settings()

    if not codebase_path:
        codebase_path = settings.codebase_path
    if not supported_languages:
        supported_languages = settings.supported_languages

    if not codebase_path:
        raise ValueError(
            "Codebase path not configured. Set CODEBASE_PATH environment variable."
        )

    if not os.path.exists(codebase_path):
        raise ValueError(f"Codebase path does not exist: {codebase_path}")

    retriever = ExactCodeRetriever(codebase_path, supported_languages)
    methods = retriever.list_all_methods()

    if not methods:
        return pd.DataFrame(columns=["method_name", "class_name", "file_path"])

    # Convert to DataFrame
    df = pd.DataFrame(methods, columns=["method_name", "class_name", "file_path"])

    logger.info(f"Found {len(methods)} methods in codebase")
    return df


def _list_all_classes_impl(
    codebase_path: str | None = None, supported_languages: list[str] | None = None
) -> pd.DataFrame:
    """Implementation function for listing all classes.

    Returns:
        pd.DataFrame: DataFrame with class information

    """
    settings = get_settings()

    if not codebase_path:
        codebase_path = settings.codebase_path
    if not supported_languages:
        supported_languages = settings.supported_languages

    if not codebase_path:
        raise ValueError(
            "Codebase path not configured. Set CODEBASE_PATH environment variable."
        )

    if not os.path.exists(codebase_path):
        raise ValueError(f"Codebase path does not exist: {codebase_path}")

    retriever = ExactCodeRetriever(codebase_path, supported_languages)
    classes = retriever.list_all_classes()

    if not classes:
        return pd.DataFrame(columns=["class_name", "file_path"])

    # Convert to DataFrame
    df = pd.DataFrame(classes, columns=["class_name", "file_path"])

    logger.info(f"Found {len(classes)} classes in codebase")
    return df


def _get_codebase_stats_impl(
    codebase_path: str | None = None, supported_languages: list[str] | None = None
) -> pd.DataFrame:
    """Implementation function for getting codebase statistics.

    Returns:
        pd.DataFrame: DataFrame with codebase statistics

    """
    from .file_loading import get_file_stats

    settings = get_settings()

    if not codebase_path:
        codebase_path = settings.codebase_path
    if not supported_languages:
        supported_languages = settings.supported_languages

    if not codebase_path:
        raise ValueError(
            "Codebase path not configured. Set CODEBASE_PATH environment variable."
        )

    if not os.path.exists(codebase_path):
        raise ValueError(f"Codebase path does not exist: {codebase_path}")

    file_list = load_files(codebase_path)
    stats = get_file_stats(file_list)

    if not stats:
        return pd.DataFrame(
            columns=["language", "file_count", "codebase_path", "supported_languages"]
        )

    # Convert to DataFrame
    data = []
    for language, count in stats.items():
        data.append(
            {
                "language": language,
                "file_count": count,
                "codebase_path": codebase_path,
                "supported_languages": ", ".join(supported_languages),
            }
        )

    df = pd.DataFrame(data)

    logger.info(
        f"Codebase contains {sum(stats.values())} files across {len(stats)} languages"
    )
    return df


# Push implementation functions to shell
_SHELL.push(
    {
        "_find_method_implementation_impl": _find_method_implementation_impl,
        "_find_class_implementation_impl": _find_class_implementation_impl,
        "_list_all_methods_impl": _list_all_methods_impl,
        "_list_all_classes_impl": _list_all_classes_impl,
        "_get_codebase_stats_impl": _get_codebase_stats_impl,
    }
)

# Conditional tool registration based on codebase path availability
if _codebase_path_available():
    logger.info("Codebase path detected - registering code retrieval tools")

    # MCP Tool implementations

    @app.tool()
    async def find_method_implementation(
        method_name: str,
        class_name: str | None = None,
        *,
        save_as: str = "method_results",
    ) -> pd.DataFrame | None:
        """Find method implementation(s) by exact name in configured programming languages.

        Args:
            method_name: The exact name of the method to find
            class_name: Optional class name to narrow down search
            save_as: Variable name to save results in IPython shell

        Returns:
            pd.DataFrame: Method implementations found

        """
        if class_name:
            code = f'{save_as} = _find_method_implementation_impl("{method_name}", "{class_name}")\n{save_as}'
        else:
            code = f'{save_as} = _find_method_implementation_impl("{method_name}")\n{save_as}'

        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def find_class_implementation(
        class_name: str, *, save_as: str = "class_results"
    ) -> pd.DataFrame | None:
        """Find class implementation(s) by exact name in configured programming languages.

        Args:
            class_name: The exact name of the class to find
            save_as: Variable name to save results in IPython shell

        Returns:
            pd.DataFrame: Class implementations found

        """
        code = f'{save_as} = _find_class_implementation_impl("{class_name}")\n{save_as}'

        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def list_all_methods(*, save_as: str = "all_methods") -> pd.DataFrame | None:
        """List all methods in the configured programming languages.

        Args:
            save_as: Variable name to save results in IPython shell

        Returns:
            pd.DataFrame: All methods with their class names and file paths

        """
        code = f"{save_as} = _list_all_methods_impl()\n{save_as}"

        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def list_all_classes(*, save_as: str = "all_classes") -> pd.DataFrame | None:
        """List all classes in the configured programming languages.

        Args:
            save_as: Variable name to save results in IPython shell

        Returns:
            pd.DataFrame: All classes with their file paths

        """
        code = f"{save_as} = _list_all_classes_impl()\n{save_as}"

        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def get_codebase_stats(
        *, save_as: str = "codebase_stats"
    ) -> pd.DataFrame | None:
        """Get statistics about the configured codebase.

        Args:
            save_as: Variable name to save results in IPython shell

        Returns:
            pd.DataFrame: Statistics about file types and counts in the codebase

        """
        code = f"{save_as} = _get_codebase_stats_impl()\n{save_as}"

        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def configure_supported_languages(
        languages: list[str], *, save_as: str = "language_config"
    ) -> str:
        """Configure which programming languages to analyze in the codebase.

        Args:
            languages: List of language names to support. Valid options: java, kotlin, python, typescript, javascript, cpp, rust
            save_as: Variable name to save configuration in IPython shell

        Returns:
            Confirmation message with the updated language configuration

        """
        valid_languages = {
            "java",
            "kotlin",
            "python",
            "typescript",
            "javascript",
            "cpp",
            "rust",
        }

        # Validate languages
        invalid_languages = []
        valid_requested = []

        for lang in languages:
            lang_lower = lang.lower().strip()
            if lang_lower in valid_languages:
                valid_requested.append(lang_lower)
            else:
                invalid_languages.append(lang)

        if invalid_languages:
            return f"Error: Invalid languages specified: {invalid_languages}. Valid options: {sorted(valid_languages)}"

        if not valid_requested:
            return "Error: No valid languages specified."

        try:
            # Save to shell variable
            config_line = f"{save_as} = {repr(valid_requested)}"
            print_line = f"print('Configured ' + str(len({save_as})) + ' languages: ' + ', '.join({save_as}))"
            code = config_line + "\n" + print_line
            await run_code_in_shell(code)

            result_msg = [
                f"Successfully configured {len(valid_requested)} languages for code analysis:",
                f"  Enabled: {', '.join(sorted(valid_requested))}",
                "",
                "Note: This configuration is for this session only.",
                "To make it permanent, set the SUPPORTED_LANGUAGES environment variable.",
                f"Example: SUPPORTED_LANGUAGES={','.join(valid_requested)}",
            ]

            return "\n".join(result_msg)
        except Exception as e:
            logger.error(f"Error configuring languages: {e}")
            return f"Error: {e}"

else:
    logger.info(
        "Codebase path not detected - code retrieval tools will not be registered"
    )

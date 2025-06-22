"""File loading utilities for code retrieval tools."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

LANGUAGE_EXTENSIONS = {
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
}


def get_language_from_extension(file_path: str) -> str:
    """Get the programming language from file extension."""
    extension = Path(file_path).suffix.lower()
    return LANGUAGE_EXTENSIONS.get(extension, "unknown")


def is_code_file(file_path: str) -> bool:
    """Check if a file is a supported code file."""
    extension = Path(file_path).suffix.lower()
    return extension in LANGUAGE_EXTENSIONS


def should_skip_directory(dir_name: str) -> bool:
    """Check if a directory should be skipped during traversal."""
    skip_dirs = {
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        "venv",
        ".venv",
        "__pycache__",
        ".idea",
        ".vscode",
        ".eclipse",
        "build",
        "dist",
        "target",
        "out",
        ".gradle",
        ".maven",
        "logs",
        "tmp",
        "temp",
    }
    return dir_name in skip_dirs


def load_files(codebase_path: str) -> list[tuple[str, str]]:
    """Load all supported code files from a codebase directory.

    Args:
        codebase_path: Path to the codebase directory

    Returns:
        List of tuples (file_path, language) for each supported code file

    """
    if not codebase_path or not os.path.exists(codebase_path):
        logger.warning(f"Codebase path does not exist: {codebase_path}")
        return []

    files = []

    try:
        for root, dirs, filenames in os.walk(codebase_path):
            dirs[:] = [d for d in dirs if not should_skip_directory(d)]

            for filename in filenames:
                file_path = os.path.join(root, filename)

                if is_code_file(file_path):
                    language = get_language_from_extension(file_path)
                    files.append((file_path, language))

    except Exception as e:
        logger.error(f"Error loading files from {codebase_path}: {e}")

    return files


def filter_files_by_language(
    files: list[tuple[str, str]], language: str
) -> list[tuple[str, str]]:
    """Filter files by programming language."""
    return [(path, lang) for path, lang in files if lang == language]


def get_file_stats(files: list[tuple[str, str]]) -> dict:
    """Get statistics about the loaded files."""
    stats = {}
    for _, language in files:
        stats[language] = stats.get(language, 0) + 1
    return stats

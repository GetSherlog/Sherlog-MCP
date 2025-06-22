"""Filesystem tools for the Sherlog MCP server.

This module provides a collection of tools for interacting with the filesystem,
inspired by the @modelcontextprotocol/server-filesystem. Operations are restricted
to a predefined list of allowed directories for security.

Design Principles:
- Data flow for list-like or tabular results primarily uses Pandas DataFrames.
- Detailed docstrings explain each tool's purpose, arguments, and output.
- Path validation is strictly enforced for all operations.
"""

import difflib
import fnmatch
import os
import pathlib
import stat
from typing import Any

import pandas as pd

from sherlog_mcp.dataframe_utils import read_csv_smart, to_pandas
from sherlog_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from sherlog_mcp.session import app, logger

ALLOWED_DIRECTORIES: list[str] = []
_allowed_dirs_env = os.environ.get("ALLOWED_DIRECTORIES", "")
if _allowed_dirs_env:
    ALLOWED_DIRECTORIES = [d.strip() for d in _allowed_dirs_env.split(",") if d.strip()]


def _filesystem_available() -> bool:
    """Return True if ALLOWED_DIRECTORIES is configured."""
    return bool(ALLOWED_DIRECTORIES)


def expand_home(filepath: str) -> str:
    """Expands a tilde (~) in a filepath to the user's home directory.

    Args:
        filepath: The path string, potentially starting with '~/' or '~'.

    Returns:
        The expanded path string.

    """
    if filepath.startswith("~/") or filepath == "~":
        return str(
            pathlib.Path.home() / filepath[2 if filepath.startswith("~/") else 1 :]
        )
    return filepath


def normalize_path(p: str) -> str:
    """Normalizes a path by resolving '..' and '.' segments and ensuring consistency.

    Args:
        p: The path string.

    Returns:
        The normalized path string.

    """
    return str(pathlib.Path(p).resolve())


async def validate_path(
    requested_path: str, check_existence: bool = True, is_for_writing: bool = False
) -> pathlib.Path:
    """Validates if the requested path is within the allowed directories and handles symlinks.

    Args:
        requested_path: The path to validate.
        check_existence: If True, the path must exist (unless is_for_writing is True and parent exists).
        is_for_writing: If True, allows the path to not exist if its parent directory is valid.

    Returns:
        A resolved and validated pathlib.Path object.

    Raises:
        PermissionError: If access is denied due to path restrictions or symlink issues.
        FileNotFoundError: If the path or its parent (for writing) does not exist.

    """
    if not ALLOWED_DIRECTORIES:
        raise PermissionError(
            "Filesystem operations are disabled: ALLOWED_DIRECTORIES is not configured."
        )

    expanded_path = expand_home(requested_path)
    
    if not os.path.isabs(expanded_path) and expanded_path != "":
        if ALLOWED_DIRECTORIES:
            if expanded_path == ".":
                expanded_path = ALLOWED_DIRECTORIES[0]
            else:
                expanded_path = os.path.join(ALLOWED_DIRECTORIES[0], expanded_path)
    
    absolute_path_str = os.path.abspath(expanded_path)
    normalized_requested = normalize_path(absolute_path_str)

    is_allowed = any(
        normalized_requested.startswith(normalize_path(allowed_dir))
        for allowed_dir in ALLOWED_DIRECTORIES
    )
    if not is_allowed:
        raise PermissionError(
            f"Access denied: Path '{absolute_path_str}' is outside the allowed directories."
        )

    p_path = pathlib.Path(absolute_path_str)

    try:
        if p_path.exists():
            real_path_str = str(p_path.resolve(strict=True))
            normalized_real = normalize_path(real_path_str)
            is_real_path_allowed = any(
                normalized_real.startswith(normalize_path(allowed_dir))
                for allowed_dir in ALLOWED_DIRECTORIES
            )
            if not is_real_path_allowed:
                raise PermissionError(
                    "Access denied: Symlink target is outside allowed directories."
                )
            final_path = pathlib.Path(real_path_str)
        elif is_for_writing:
            parent_dir = p_path.parent
            if not parent_dir.exists():
                raise FileNotFoundError(
                    f"Parent directory does not exist: {str(parent_dir)}"
                )
            real_parent_path_str = str(parent_dir.resolve(strict=True))
            normalized_parent = normalize_path(real_parent_path_str)
            is_parent_allowed = any(
                normalized_parent.startswith(normalize_path(allowed_dir))
                for allowed_dir in ALLOWED_DIRECTORIES
            )
            if not is_parent_allowed:
                raise PermissionError(
                    "Access denied: Parent directory for writing is outside allowed directories."
                )
            final_path = p_path
        elif check_existence:
            raise FileNotFoundError(f"Path does not exist: {str(p_path)}")
        else:
            final_path = p_path

    except FileNotFoundError as e:
        if is_for_writing:
            parent_dir = p_path.parent
            if not parent_dir.exists():
                raise FileNotFoundError(
                    f"Parent directory does not exist: {str(parent_dir)}"
                ) from e
            real_parent_path_str = str(parent_dir.resolve(strict=True))
            normalized_parent = normalize_path(real_parent_path_str)
            is_parent_allowed = any(
                normalized_parent.startswith(normalize_path(allowed_dir))
                for allowed_dir in ALLOWED_DIRECTORIES
            )
            if not is_parent_allowed:
                raise PermissionError(
                    "Access denied: Parent directory for writing is outside allowed directories."
                ) from e
            final_path = p_path
        elif check_existence:
            raise FileNotFoundError(
                f"Path does not exist or is invalid: {str(p_path)}"
            ) from e
        else:
            final_path = p_path

    return final_path


def _normalize_line_endings(text: str) -> str:
    """Normalizes line endings to \\n."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _create_unified_diff(
    original_content: str, new_content: str, filepath: str = "file"
) -> str:
    """Creates a unified diff string."""
    normalized_original = _normalize_line_endings(original_content)
    normalized_new = _normalize_line_endings(new_content)
    diff = difflib.unified_diff(
        normalized_original.splitlines(keepends=True),
        normalized_new.splitlines(keepends=True),
        fromfile=filepath,
        tofile=filepath,
        lineterm="\\n",
    )
    return "".join(diff)


async def _build_tree_recursive(
    current_path: pathlib.Path, root_validated_path_str: str
) -> list[dict[str, Any]]:
    """Helper to recursively build directory tree, ensuring sub-paths are validated."""
    tree_entries = []
    for entry in current_path.iterdir():
        entry_data: dict[str, Any] = {"name": entry.name}
        try:
            resolved_entry_path = entry.resolve()
            await validate_path(str(resolved_entry_path), check_existence=True)

            if entry.is_dir():
                entry_data["type"] = "directory"
                entry_data["children"] = await _build_tree_recursive(
                    resolved_entry_path, root_validated_path_str
                )
            else:
                entry_data["type"] = "file"
            tree_entries.append(entry_data)
        except (PermissionError, FileNotFoundError):
            entry_data["type"] = "inaccessible"
            tree_entries.append(entry_data)
        except Exception:
            entry_data["type"] = "error"
            tree_entries.append(entry_data)

    return tree_entries


if _filesystem_available():
    async def _read_file_impl(file_path: str) -> pd.DataFrame:
        valid = await validate_path(file_path)
        df = read_csv_smart(str(valid))
        return to_pandas(df)

    async def _list_directory_impl(path: str) -> pd.DataFrame | None:
        valid_path = await validate_path(path)
        if not valid_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {valid_path}")

        entries = []
        try:
            for entry in valid_path.iterdir():
                try:
                    await validate_path(str(entry.resolve()), check_existence=True)
                    entry_type = "directory" if entry.is_dir() else "file"
                    entries.append(
                        {
                            "name": entry.name,
                            "type": entry_type,
                            "path": str(entry.resolve()),
                        }
                    )
                except (PermissionError, FileNotFoundError):
                    entries.append(
                        {"name": entry.name, "type": "inaccessible", "path": str(entry)}
                    )

        except Exception as e:
            raise OSError(f"Error listing directory {valid_path}: {e}")

        if not entries:
            return pd.DataFrame(columns=["name", "type", "path"])
        return pd.DataFrame(entries)

    async def _directory_tree_impl(path: str) -> pd.DataFrame | None:
        valid_path = await validate_path(path)
        if not valid_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {valid_path}")

        tree_structure = await _build_tree_recursive(valid_path, str(valid_path))
        return pd.DataFrame(tree_structure)

    async def _search_files_impl(
        path: str, pattern: str, exclude_patterns: list[str], recursive: bool
    ) -> pd.DataFrame | None:
        root_path_str = path
        valid_root_path = await validate_path(root_path_str)
        if not valid_root_path.is_dir():
            raise NotADirectoryError(
                f"Root path for search is not a directory: {valid_root_path}"
            )

        results: list[str] = []

        for dirpath_str, dirnames, filenames in os.walk(
            str(valid_root_path), topdown=True
        ):
            current_dir_path = pathlib.Path(dirpath_str)
            if exclude_patterns:
                original_dirnames = list(dirnames)
                dirnames[:] = [
                    d
                    for d in original_dirnames
                    if not any(
                        fnmatch.fnmatch(d, ep)
                        or fnmatch.fnmatch(str(current_dir_path / d), ep)
                        for ep in exclude_patterns
                    )
                ]

            all_entries = filenames + dirnames

            for entry_name in all_entries:
                full_entry_path = current_dir_path / entry_name

                try:
                    await validate_path(
                        str(full_entry_path.resolve()), check_existence=True
                    )

                    is_excluded = False
                    if exclude_patterns:
                        relative_path_to_root = str(
                            full_entry_path.relative_to(valid_root_path)
                        )
                        if any(
                            fnmatch.fnmatch(entry_name, ep)
                            or fnmatch.fnmatch(relative_path_to_root, ep)
                            or fnmatch.fnmatch(str(full_entry_path), ep)
                            for ep in exclude_patterns
                        ):
                            is_excluded = True

                    if not is_excluded and fnmatch.fnmatch(entry_name, pattern):
                        results.append(str(full_entry_path.resolve()))

                except (PermissionError, FileNotFoundError):
                    continue
                except Exception:
                    continue

            if not recursive:
                break

        if not results:
            return pd.DataFrame(columns=["path"])
        return pd.DataFrame(results, columns=["path"])

    async def _get_file_info_impl(path: str) -> pd.DataFrame | None:
        valid_path = await validate_path(path)
        try:
            stats = valid_path.stat()
            info_dict = {
                "name": valid_path.name,
                "path": str(valid_path),
                "type": "directory" if valid_path.is_dir() else "file",
                "size_bytes": stats.st_size,
                "created_timestamp": stats.st_birthtime
                if hasattr(stats, "st_birthtime")
                else stats.st_ctime,
                "modified_timestamp": stats.st_mtime,
                "accessed_timestamp": stats.st_atime,
                "permissions_octal": oct(stat.S_IMODE(stats.st_mode))[-3:],
                "is_symlink": valid_path.is_symlink(),
                "absolute_path": str(valid_path.resolve()),
                "symlink_target": None,
            }
            if info_dict["is_symlink"]:
                try:
                    info_dict["symlink_target"] = str(valid_path.readlink())
                except Exception:
                    info_dict["symlink_target"] = "Error reading symlink target"

            return pd.DataFrame(info_dict)

        except Exception as e:
            raise OSError(f"Error getting file info for {valid_path}: {e}")

    def _list_allowed_directories_impl() -> pd.DataFrame:
        if not ALLOWED_DIRECTORIES:
            return pd.DataFrame(columns=["allowed_directory_path"])

        normalized_allowed_dirs = [
            normalize_path(expand_home(d)) for d in ALLOWED_DIRECTORIES
        ]
        
        result_df = pd.DataFrame(normalized_allowed_dirs, columns=["allowed_directory_path"])
        return result_df

    _SHELL.push(
        {
            "_normalize_line_endings": _normalize_line_endings,
            "_create_unified_diff": _create_unified_diff,
            "_validate_path": validate_path,
            "_list_directory_impl": _list_directory_impl,
            "_read_file_impl": _read_file_impl,
            "_build_tree_recursive": _build_tree_recursive,
            "_directory_tree_impl": _directory_tree_impl,
            "_search_files_impl": _search_files_impl,
            "_get_file_info_impl": _get_file_info_impl,
            "_list_allowed_directories_impl": _list_allowed_directories_impl,
        }
    )

    @app.tool()
    async def read_file(file_path: str, save_as: str) -> pd.DataFrame | None:
        """Load a CSV file into a DataFrame stored as *save_as* in the shell.
        You can then use this DataFrame (defined by *save_as*) in subsequent steps.
        
        Args:
            file_path (str): Path to the CSV file to read
            save_as (str): Variable name to store the DataFrame
            
        Returns:
            pd.DataFrame: The loaded CSV data
            
        Examples
        --------
        After calling this tool with save_as="df":
        
        # View the loaded data
        >>> execute_python_code("df")
        
        # Check shape and columns
        >>> execute_python_code("df.shape")
        >>> execute_python_code("df.columns.tolist()")
        
        # View data types and info
        >>> execute_python_code("df.info()")
        >>> execute_python_code("df.dtypes")
        
        # Basic statistics
        >>> execute_python_code("df.describe()")
        
        # Filter data
        >>> execute_python_code("df[df['column_name'] > 100]")
        
        # Check for missing values
        >>> execute_python_code("df.isnull().sum()")
        """
        code = f'{save_as} = await _read_file_impl("{file_path}")'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def list_directory(dir_path: str, save_as: str) -> pd.DataFrame | None:
        """Gets a detailed listing of all files and directories in a specified path.

        Args:
            dir_path: The path to the directory to list.
            save_as: The name of the variable to save the DataFrame to in the shell.

        Returns:
            A Pandas DataFrame with columns: 'name', 'type' ('file' or 'directory'), 'path'.
            
        Examples
        --------
        After calling this tool with save_as="files":
        
        # View all items in directory
        >>> execute_python_code("files")
        
        # Filter for files only
        >>> execute_python_code("files[files['type'] == 'file']")
        
        # Filter for directories only
        >>> execute_python_code("files[files['type'] == 'directory']")
        
        # Get file names matching pattern
        >>> execute_python_code("files[files['name'].str.endswith('.py')]")
        
        # Sort by name
        >>> execute_python_code("files.sort_values('name')")
        
        # Count files vs directories
        >>> execute_python_code("files['type'].value_counts()")
        
        # Get full paths as list
        >>> execute_python_code("files['path'].tolist()")

        """
        code = f'{save_as} = await _list_directory_impl("{dir_path}")\n{save_as}'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def directory_tree(path: str, save_as: str) -> pd.DataFrame | None:
        """Gets a recursive tree view of files and directories as a JSON string.

        Args:
            path (str): The root path to generate tree from
            save_as (str): Variable name to store the tree structure

        Returns:
            str: A JSON string representing the directory tree.
            Each entry includes 'name', 'type' ('file'/'directory'/'inaccessible'/'error'),
            and 'children' for directories.
            
        Examples
        --------
        After calling this tool with save_as="tree":
        
        # View the tree structure
        >>> execute_python_code("print(tree[:1000])")
        
        # Parse as JSON for analysis
        >>> execute_python_code("import json; tree_data = json.loads(tree)")
        
        # Pretty print the tree
        >>> execute_python_code("import json; print(json.dumps(json.loads(tree), indent=2)[:2000])")
        
        # Count total items
        >>> execute_python_code("tree.count('\"name\"')")
        
        # Find specific files
        >>> execute_python_code("'README.md' in tree")

        """
        code = f'{save_as} = await _directory_tree_impl("{path}")\n{save_as}'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def search_files(
        path: str,
        pattern: str,
        exclude_patterns: list[str],
        recursive: bool,
        save_as: str,
    ) -> pd.DataFrame | None:
        """Recursively searches for files and directories matching a pattern.

        Args:
            path (str): Root directory to search in
            pattern (str): File pattern to match (supports wildcards)
            exclude_patterns (list[str]): Patterns to exclude from results
            recursive (bool): Whether to search subdirectories
            save_as (str): Variable name to store search results

        Returns:
            pd.DataFrame: DataFrame with a single column 'path' containing full paths to matching items.
            Returns an empty DataFrame if no matches are found.
            
        Examples
        --------
        After calling this tool with save_as="matches":
        
        # View all matching files
        >>> execute_python_code("matches")
        
        # Count matches
        >>> execute_python_code("len(matches)")
        
        # Get file names only (without path)
        >>> execute_python_code("matches['path'].apply(lambda x: x.split('/')[-1])")
        
        # Filter by additional criteria
        >>> execute_python_code("matches[matches['path'].str.contains('test')]")
        
        # Convert to list for iteration
        >>> execute_python_code("matches['path'].tolist()")
        
        # Check if specific file was found
        >>> execute_python_code("any('config.json' in path for path in matches['path'])")

        """
        code = f'{save_as} = await _search_files_impl("{path}", "{pattern}", "{exclude_patterns}", "{recursive}")\n{save_as}'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def get_file_info(path: str, save_as: str) -> pd.DataFrame | None:
        """Retrieves detailed metadata about a file or directory.

        Args:
            path (str): Path to the file or directory
            save_as (str): Variable name to store file info

        Returns:
            pd.DataFrame: DataFrame with a single row containing metadata:
            'name', 'path', 'type', 'size_bytes', 'created_timestamp',
            'modified_timestamp', 'accessed_timestamp', 'permissions_octal',
            'is_symlink', 'absolute_path', and 'symlink_target' (if applicable).
            
        Examples
        --------
        After calling this tool with save_as="info":
        
        # View all file metadata
        >>> execute_python_code("info")
        
        # Get specific attributes
        >>> execute_python_code("info.iloc[0]['size_bytes']")
        >>> execute_python_code("info.iloc[0]['type']")
        
        # Convert to dictionary for easy access
        >>> execute_python_code("info.iloc[0].to_dict()")
        
        # Check file size in MB
        >>> execute_python_code("info.iloc[0]['size_bytes'] / (1024 * 1024)")
        
        # Format timestamps
        >>> execute_python_code("import pandas as pd; pd.to_datetime(info.iloc[0]['modified_timestamp'])")
        
        # Check permissions
        >>> execute_python_code("oct(info.iloc[0]['permissions_octal'])")

        """
        code = f'{save_as} = await _get_file_info_impl("{path}")\n{save_as}'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def list_allowed_directories() -> pd.DataFrame | None:
        """Returns the list of directories that this server is allowed to access.

        Args:
            args: ListAllowedDirectoriesArgs (empty).

        Returns:
            A Pandas DataFrame with a single column 'allowed_directory_path'.

        """
        code = "_mcp_allowed_dirs_df = _list_allowed_directories_impl()\n_mcp_allowed_dirs_df"
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    def peek_file(file_path: str, n_lines: int = 10) -> list[str]:
        """Return the first `n_lines` of a text file without fully loading or parsing it.

        This tool is useful for a quick inspection of a file's content, for example,
        to understand its structure, check for headers, or get a glimpse of the data
        before deciding how to process it with other tools (e.g., `load_file_log_data`).

        Parameters
        ----------
        file_path : str
            The absolute or relative path to the text file that needs to be peeked into.
            This argument is **not** resolved from `session_vars`; it must be a
            string literal representing the file path.
        n_lines : int, default 10
            The number of lines to read from the beginning of the file.
            If the file has fewer than `n_lines`, all lines in the file are returned.

        Returns
        -------
        List[str]
            A list of strings, where each string is a line read from the file,
            with trailing newline characters (e.g., '\\n', '\\r\\n') removed.
            If the file cannot be opened or read (e.g., file not found, permission
            denied), an exception will be raised by the underlying file operation.

        Examples
        --------
        >>> peek_file(file_path="data.csv", n_lines=5)
        ['header1,header2,header3', 'val1,val2,val3', ...]

        >>> peek_file(file_path="short_file.txt", n_lines=20)

        See Also
        --------
        load_file_log_data : For loading and parsing structured log data from files.
        get_log_file_columns : To get column names from a CSV file.

        Notes
        -----
        - The file is opened in read mode ('r') with UTF-8 encoding. Errors during
          decoding are replaced with a replacement character.
        - This tool does not store any data in `session_vars`.
        - It's designed for text files. Peeking into binary files might produce
          unreadable output.

        """
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            return [next(fh).rstrip("\\n") for _ in range(n_lines)]

else:
    logger.info(
        "ALLOWED_DIRECTORIES not configured - filesystem tools will not be registered"
    )

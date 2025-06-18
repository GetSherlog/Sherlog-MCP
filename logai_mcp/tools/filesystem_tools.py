"""
Filesystem tools for the LogAI MCP server.

This module provides a collection of tools for interacting with the filesystem,
inspired by the @modelcontextprotocol/server-filesystem. Operations are restricted
to a predefined list of allowed directories for security.

Design Principles:
- Data flow for list-like or tabular results primarily uses Pandas DataFrames.
- Detailed docstrings explain each tool's purpose, arguments, and output.
- Path validation is strictly enforced for all operations.
"""

import os
import pathlib
import stat
import difflib
import fnmatch
from typing import List, Dict, Any, Optional

import pandas as pd
from logai_mcp.dataframe_utils import read_csv_smart, to_pandas

from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.session import app, logger

# Initialize ALLOWED_DIRECTORIES from environment variable
ALLOWED_DIRECTORIES: List[str] = []
_allowed_dirs_env = os.environ.get('ALLOWED_DIRECTORIES', '')
if _allowed_dirs_env:
    ALLOWED_DIRECTORIES = [d.strip() for d in _allowed_dirs_env.split(',') if d.strip()]


def _filesystem_available() -> bool:
    """Return True if ALLOWED_DIRECTORIES is configured."""
    return bool(ALLOWED_DIRECTORIES)


def expand_home(filepath: str) -> str:
    """
    Expands a tilde (~) in a filepath to the user's home directory.

    Args:
        filepath: The path string, potentially starting with '~/' or '~'.

    Returns:
        The expanded path string.
    """
    if filepath.startswith('~/') or filepath == '~':
        return str(pathlib.Path.home() / filepath[2 if filepath.startswith('~/') else 1:])
    return filepath

def normalize_path(p: str) -> str:
    """
    Normalizes a path by resolving '..' and '.' segments and ensuring consistency.

    Args:
        p: The path string.

    Returns:
        The normalized path string.
    """
    return str(pathlib.Path(p).resolve())

async def validate_path(requested_path: str, check_existence: bool = True, is_for_writing: bool = False) -> pathlib.Path:
    """
    Validates if the requested path is within the allowed directories and handles symlinks.

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
        raise PermissionError("Filesystem operations are disabled: ALLOWED_DIRECTORIES is not configured.")

    expanded_path = expand_home(requested_path)
    absolute_path_str = os.path.abspath(expanded_path)
    normalized_requested = normalize_path(absolute_path_str)
    
    is_allowed = any(normalized_requested.startswith(normalize_path(allowed_dir)) for allowed_dir in ALLOWED_DIRECTORIES)
    if not is_allowed:
        raise PermissionError(
            f"Access denied: Path '{absolute_path_str}' is outside the allowed directories."
        )

    p_path = pathlib.Path(absolute_path_str)

    try:
        if p_path.exists():
            real_path_str = str(p_path.resolve(strict=True))
            normalized_real = normalize_path(real_path_str)
            is_real_path_allowed = any(normalized_real.startswith(normalize_path(allowed_dir)) for allowed_dir in ALLOWED_DIRECTORIES)
            if not is_real_path_allowed:
                raise PermissionError("Access denied: Symlink target is outside allowed directories.")
            final_path = pathlib.Path(real_path_str)
        elif is_for_writing:
            parent_dir = p_path.parent
            if not parent_dir.exists():
                 raise FileNotFoundError(f"Parent directory does not exist: {str(parent_dir)}")
            real_parent_path_str = str(parent_dir.resolve(strict=True))
            normalized_parent = normalize_path(real_parent_path_str)
            is_parent_allowed = any(normalized_parent.startswith(normalize_path(allowed_dir)) for allowed_dir in ALLOWED_DIRECTORIES)
            if not is_parent_allowed:
                raise PermissionError("Access denied: Parent directory for writing is outside allowed directories.")
            final_path = p_path
        elif check_existence:
            raise FileNotFoundError(f"Path does not exist: {str(p_path)}")
        else:
            final_path = p_path


    except FileNotFoundError as e:
        if is_for_writing:
             parent_dir = p_path.parent
             if not parent_dir.exists():
                 raise FileNotFoundError(f"Parent directory does not exist: {str(parent_dir)}") from e
             real_parent_path_str = str(parent_dir.resolve(strict=True))
             normalized_parent = normalize_path(real_parent_path_str)
             is_parent_allowed = any(normalized_parent.startswith(normalize_path(allowed_dir)) for allowed_dir in ALLOWED_DIRECTORIES)
             if not is_parent_allowed:
                 raise PermissionError("Access denied: Parent directory for writing is outside allowed directories.") from e
             final_path = p_path
        elif check_existence:
            raise FileNotFoundError(f"Path does not exist or is invalid: {str(p_path)}") from e
        else:
            final_path = p_path


    return final_path

def _normalize_line_endings(text: str) -> str:
    """Normalizes line endings to \\n."""
    return text.replace('\r\n', '\n').replace('\r', '\n')

def _create_unified_diff(original_content: str, new_content: str, filepath: str = 'file') -> str:
    """Creates a unified diff string."""
    normalized_original = _normalize_line_endings(original_content)
    normalized_new = _normalize_line_endings(new_content)
    diff = difflib.unified_diff(
        normalized_original.splitlines(keepends=True),
        normalized_new.splitlines(keepends=True),
        fromfile=filepath,
        tofile=filepath,
        lineterm='\\n'
    )
    return "".join(diff)


async def _build_tree_recursive(current_path: pathlib.Path, root_validated_path_str: str) -> List[Dict[str, Any]]:
    """Helper to recursively build directory tree, ensuring sub-paths are validated."""
    tree_entries = []
    for entry in current_path.iterdir():
        entry_data: Dict[str, Any] = {'name': entry.name}
        try:
            resolved_entry_path = entry.resolve()
            await validate_path(str(resolved_entry_path), check_existence=True)

            if entry.is_dir():
                entry_data['type'] = 'directory'
                entry_data['children'] = await _build_tree_recursive(resolved_entry_path, root_validated_path_str)
            else:
                entry_data['type'] = 'file'
            tree_entries.append(entry_data)
        except (PermissionError, FileNotFoundError):
            entry_data['type'] = 'inaccessible'
            tree_entries.append(entry_data)
        except Exception:
            entry_data['type'] = 'error'
            tree_entries.append(entry_data)
            
    return tree_entries


if _filesystem_available():
    logger.info(f"ALLOWED_DIRECTORIES configured ({len(ALLOWED_DIRECTORIES)} directories) - registering filesystem tools")

    async def _read_file_impl(file_path: str) -> pd.DataFrame:
        valid = await validate_path(file_path)
        df = read_csv_smart(str(valid))
        return to_pandas(df)

    async def _list_directory_impl(path: str) -> Optional[pd.DataFrame]:
        valid_path = await validate_path(path)
        if not valid_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {valid_path}")

        entries = []
        try:
            for entry in valid_path.iterdir():
                try:
                    await validate_path(str(entry.resolve()), check_existence=True)
                    entry_type = 'directory' if entry.is_dir() else 'file'
                    entries.append({'name': entry.name, 'type': entry_type, 'path': str(entry.resolve())})
                except (PermissionError, FileNotFoundError):
                    entries.append({'name': entry.name, 'type': 'inaccessible', 'path': str(entry)})
                    
        except Exception as e:
            raise IOError(f"Error listing directory {valid_path}: {e}")
        
        if not entries:
            return pd.DataFrame(columns=['name', 'type', 'path'])
        return pd.DataFrame(entries)


    async def _directory_tree_impl(path: str) -> Optional[pd.DataFrame]:
        valid_path = await validate_path(path)
        if not valid_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {valid_path}")

        tree_structure = await _build_tree_recursive(valid_path, str(valid_path))
        return pd.DataFrame(tree_structure)


    async def _search_files_impl(path: str, pattern: str, exclude_patterns: List[str], recursive: bool) -> Optional[pd.DataFrame]:
        root_path_str = path
        valid_root_path = await validate_path(root_path_str)
        if not valid_root_path.is_dir():
            raise NotADirectoryError(f"Root path for search is not a directory: {valid_root_path}")

        results: List[str] = []
        
        for dirpath_str, dirnames, filenames in os.walk(str(valid_root_path), topdown=True):
            current_dir_path = pathlib.Path(dirpath_str)
            if exclude_patterns:
                original_dirnames = list(dirnames)
                dirnames[:] = [d for d in original_dirnames if not any(fnmatch.fnmatch(d, ep) or fnmatch.fnmatch(str(current_dir_path/d), ep) for ep in exclude_patterns)]

            all_entries = filenames + dirnames
            
            for entry_name in all_entries:
                full_entry_path = current_dir_path / entry_name
                
                try:
                    await validate_path(str(full_entry_path.resolve()), check_existence=True)

                    is_excluded = False
                    if exclude_patterns:
                        relative_path_to_root = str(full_entry_path.relative_to(valid_root_path))
                        if any(fnmatch.fnmatch(entry_name, ep) or \
                               fnmatch.fnmatch(relative_path_to_root, ep) or \
                               fnmatch.fnmatch(str(full_entry_path), ep) for ep in exclude_patterns):
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
            return pd.DataFrame(columns=['path'])
        return pd.DataFrame(results, columns=['path'])


    async def _get_file_info_impl(path: str) -> Optional[pd.DataFrame]:
        valid_path = await validate_path(path)
        try:
            stats = valid_path.stat()
            info_dict = {
                "name": valid_path.name,
                "path": str(valid_path),
                "type": "directory" if valid_path.is_dir() else "file",
                "size_bytes": stats.st_size,
                "created_timestamp": stats.st_birthtime if hasattr(stats, 'st_birthtime') else stats.st_ctime,
                "modified_timestamp": stats.st_mtime,
                "accessed_timestamp": stats.st_atime,
                "permissions_octal": oct(stat.S_IMODE(stats.st_mode))[-3:],
                "is_symlink": valid_path.is_symlink(),
                "absolute_path": str(valid_path.resolve()),
                "symlink_target": None  # Initialize symlink_target
            }
            if info_dict["is_symlink"]:
                try:
                    info_dict["symlink_target"] = str(valid_path.readlink())
                except Exception:
                    info_dict["symlink_target"] = "Error reading symlink target"
            
            return pd.DataFrame(info_dict)

        except Exception as e:
            raise IOError(f"Error getting file info for {valid_path}: {e}")
        

    def _list_allowed_directories_impl() -> pd.DataFrame:
        if not ALLOWED_DIRECTORIES:
            return pd.DataFrame(columns=['allowed_directory_path'])
        
        normalized_allowed_dirs = [normalize_path(expand_home(d)) for d in ALLOWED_DIRECTORIES]
        
        return pd.DataFrame(normalized_allowed_dirs, columns=['allowed_directory_path'])


    _SHELL.push({"_normalize_line_endings": _normalize_line_endings, 
                 "_create_unified_diff": _create_unified_diff, 
                 "_validate_path": validate_path,
                 "_list_directory_impl": _list_directory_impl,
                 "_read_file_impl": _read_file_impl,
                 "_build_tree_recursive": _build_tree_recursive,
                 "_directory_tree_impl": _directory_tree_impl,
                 "_search_files_impl": _search_files_impl,
                 "_get_file_info_impl": _get_file_info_impl,
                 "_list_allowed_directories_impl": _list_allowed_directories_impl,
                })


    @app.tool()
    async def read_file(file_path: str, save_as: str) -> Optional[pd.DataFrame]:
        """
        Load a CSV file into a DataFrame stored as *save_as* in the shell. 
        You can then use this DataFrame (defined by *save_as*) in subsequent steps.
        """

        code = f"{save_as} = await _read_file_impl(\"{file_path}\")"
        return await run_code_in_shell(code)


    @app.tool()
    async def list_directory(dir_path: str, save_as: str) -> Optional[pd.DataFrame]:
        """
        Gets a detailed listing of all files and directories in a specified path.

        Args:
            dir_path: The path to the directory to list.
            save_as: The name of the variable to save the DataFrame to in the shell.

        Returns:
            A Pandas DataFrame with columns: 'name', 'type' ('file' or 'directory'), 'path'.
        """
        
        code = f"{save_as} = await _list_directory_impl(\"{dir_path}\")"
        return await run_code_in_shell(code)

    @app.tool()
    async def directory_tree(path: str, save_as: str) -> Optional[pd.DataFrame]:
        """
        Gets a recursive tree view of files and directories as a JSON string.

        Args:
            args: A DirectoryTreeArgs object containing the root path.

        Returns:
            A JSON string representing the directory tree.
            Each entry includes 'name', 'type' ('file'/'directory'/'inaccessible'/'error'),
            and 'children' for directories.
        """
        
        code = f"{save_as} = await _directory_tree_impl(\"{path}\")"
        return await run_code_in_shell(code)


    @app.tool()
    async def search_files(path: str, pattern: str, exclude_patterns: List[str], recursive: bool, save_as: str) -> Optional[pd.DataFrame]:
        """
        Recursively searches for files and directories matching a pattern.

        Args:
            args: A SearchFilesArgs object with path, pattern, exclude_patterns, and recursive flag.

        Returns:
            A Pandas DataFrame with a single column 'path' containing full paths to matching items.
            Returns an empty DataFrame if no matches are found.
        """

        code = f"{save_as} = await _search_files_impl(\"{path}\", \"{pattern}\", \"{exclude_patterns}\", \"{recursive}\")"
        return await run_code_in_shell(code)
        

    @app.tool()
    async def get_file_info(path: str, save_as: str) -> Optional[pd.DataFrame]:
        """
        Retrieves detailed metadata about a file or directory.

        Args:
            args: A GetFileInfoArgs object containing the path.

        Returns:
            A Pandas DataFrame with a single row and columns for each piece of metadata:
            'name', 'path', 'type', 'size_bytes', 'created_timestamp',
            'modified_timestamp', 'accessed_timestamp', 'permissions_octal',
            'is_symlink', 'absolute_path', and 'symlink_target' (if applicable).
        """

        code = f"{save_as} = await _get_file_info_impl(\"{path}\")"
        return await run_code_in_shell(code)


    @app.tool()
    async def list_allowed_directories() -> Optional[pd.DataFrame]:
        """
        Returns the list of directories that this server is allowed to access.

        Args:
            args: ListAllowedDirectoriesArgs (empty).

        Returns:
            A Pandas DataFrame with a single column 'allowed_directory_path'.
        """

        return await run_code_in_shell("_list_allowed_directories_impl()")


    @app.tool()
    def peek_file(file_path: str, n_lines: int = 10) -> List[str]:
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
        # Peek at the first 5 lines of 'data.csv'
        >>> peek_file(file_path="data.csv", n_lines=5)
        ['header1,header2,header3', 'val1,val2,val3', ...] # Example output

        # Peek at a file with fewer than n_lines
        >>> peek_file(file_path="short_file.txt", n_lines=20)
        # If short_file.txt has only 3 lines, it returns those 3 lines.

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
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            return [next(fh).rstrip("\\n") for _ in range(n_lines)]

else:
    logger.info("ALLOWED_DIRECTORIES not configured - filesystem tools will not be registered") 
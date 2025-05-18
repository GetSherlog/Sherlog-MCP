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

import asyncio
import os
import pathlib
import shutil
import stat
import json
import difflib
import fnmatch
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, validator

from logai_mcp.session import app, log_tool # Added import

# --- Configuration ---
# This would typically be loaded from server configuration or environment variables.
# For now, it's a placeholder. It MUST be populated with absolute, normalized paths.
ALLOWED_DIRECTORIES: List[str] = []

# --- Helper Functions ---

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
    # Use os.path.abspath to handle relative paths correctly before normalization
    absolute_path_str = os.path.abspath(expanded_path)
    normalized_requested = normalize_path(absolute_path_str)
    
    # Check if path is within allowed directories
    is_allowed = any(normalized_requested.startswith(normalize_path(allowed_dir)) for allowed_dir in ALLOWED_DIRECTORIES)
    if not is_allowed:
        raise PermissionError(
            f"Access denied: Path '{absolute_path_str}' is outside the allowed directories."
        )

    p_path = pathlib.Path(absolute_path_str)

    try:
        # Handle symlinks by checking their real path
        if p_path.exists(): # Realpath only works for existing paths
            real_path_str = str(p_path.resolve(strict=True)) # Resolves symlinks
            normalized_real = normalize_path(real_path_str)
            is_real_path_allowed = any(normalized_real.startswith(normalize_path(allowed_dir)) for allowed_dir in ALLOWED_DIRECTORIES)
            if not is_real_path_allowed:
                raise PermissionError("Access denied: Symlink target is outside allowed directories.")
            final_path = pathlib.Path(real_path_str)
        elif is_for_writing:
            # For new files/dirs, verify parent directory
            parent_dir = p_path.parent
            if not parent_dir.exists():
                 raise FileNotFoundError(f"Parent directory does not exist: {str(parent_dir)}")
            real_parent_path_str = str(parent_dir.resolve(strict=True))
            normalized_parent = normalize_path(real_parent_path_str)
            is_parent_allowed = any(normalized_parent.startswith(normalize_path(allowed_dir)) for allowed_dir in ALLOWED_DIRECTORIES)
            if not is_parent_allowed:
                raise PermissionError("Access denied: Parent directory for writing is outside allowed directories.")
            final_path = p_path # Path doesn't exist yet, but parent is okay
        elif check_existence:
            # Path does not exist and we are not writing to it
            raise FileNotFoundError(f"Path does not exist: {str(p_path)}")
        else:
            # Path does not exist, but existence check is off (e.g. for move destination)
            final_path = p_path


    except FileNotFoundError as e:
        # This can happen if strict=True and path doesn't exist or part of it is a broken symlink
        if is_for_writing: # If we are writing, and parent check passed, this is okay
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
        lineterm='\\n' # Ensure diff uses \n
    )
    return "".join(diff)

# --- Pydantic Schemas for Tool Arguments ---

class ReadFileArgs(BaseModel):
    path: str = Field(..., description="The path to the file to read.")

class ReadMultipleFilesArgs(BaseModel):
    paths: List[str] = Field(..., description="A list of paths to the files to read.")

class WriteFileArgs(BaseModel):
    path: str = Field(..., description="The path to the file to write.")
    content: str = Field(..., description="The content to write to the file.")

class EditOperation(BaseModel):
    old_text: str = Field(..., description="Text to search for. Must match exactly the content in the file, including line endings and indentation.")
    new_text: str = Field(..., description="Text to replace with.")

class EditFileArgs(BaseModel):
    path: str = Field(..., description="The path to the file to edit.")
    edits: List[EditOperation] = Field(..., description="A list of edit operations to perform.")
    dry_run: bool = Field(False, description="If True, preview changes using git-style diff format without writing to disk.")

class CreateDirectoryArgs(BaseModel):
    path: str = Field(..., description="The path to the directory to create. Parent directories will be created if they don't exist.")

class ListDirectoryArgs(BaseModel):
    path: str = Field(..., description="The path to the directory to list.")

class DirectoryTreeArgs(BaseModel):
    path: str = Field(..., description="The path to the root directory for the tree view.")

class MoveFileArgs(BaseModel):
    source: str = Field(..., description="The path to the source file or directory.")
    destination: str = Field(..., description="The path to the destination.")

class SearchFilesArgs(BaseModel):
    path: str = Field(..., description="The root directory to start the search from.")
    pattern: str = Field(..., description="The search pattern (e.g., '*.txt', 'report_*.csv'). Uses fnmatch-style globbing.")
    exclude_patterns: Optional[List[str]] = Field(default_factory=list, description="A list of patterns to exclude from the search (e.g., '*/.git/*', '*.log').")
    recursive: bool = Field(True, description="If True, search recursively into subdirectories.")

class GetFileInfoArgs(BaseModel):
    path: str = Field(..., description="The path to the file or directory to get information about.")

class ListAllowedDirectoriesArgs(BaseModel):
    pass # No arguments needed


# --- Tool Implementations ---

@app.tool()
@log_tool
async def read_file(args: ReadFileArgs) -> str:
    """
    Reads the complete contents of a file from the file system.

    Args:
        args: A ReadFileArgs object containing the path to the file.

    Returns:
        The content of the file as a string.

    Raises:
        PermissionError: If path validation fails.
        FileNotFoundError: If the file does not exist.
        IOError: If any other reading error occurs.
    """
    valid_path = await validate_path(args.path)
    try:
        with open(valid_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        raise IOError(f"Error reading file {valid_path}: {e}")

@app.tool()
@log_tool
async def read_multiple_files(args: ReadMultipleFilesArgs) -> pd.DataFrame:
    """
    Reads the contents of multiple files simultaneously.

    Args:
        args: A ReadMultipleFilesArgs object containing a list of file paths.

    Returns:
        A Pandas DataFrame with columns: 'path', 'content', 'error'.
        'content' will be the file content if successful, None otherwise.
        'error' will be the error message if reading failed, None otherwise.
    """
    results = []
    for file_path_str in args.paths:
        content = None
        error_message = None
        try:
            valid_path = await validate_path(file_path_str)
            with open(valid_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            error_message = str(e)
        results.append({'path': file_path_str, 'content': content, 'error': error_message})
    return pd.DataFrame(results)

@app.tool()
@log_tool
async def write_file(args: WriteFileArgs) -> str:
    """
    Creates a new file or completely overwrites an existing file with new content.

    Args:
        args: A WriteFileArgs object containing the path and content.

    Returns:
        A success message string.

    Raises:
        PermissionError: If path validation fails.
        IOError: If any writing error occurs.
    """
    valid_path = await validate_path(args.path, check_existence=False, is_for_writing=True)
    try:
        # Ensure parent directory exists
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        with open(valid_path, 'w', encoding='utf-8') as f:
            f.write(args.content)
        return f"Successfully wrote to {valid_path}"
    except Exception as e:
        raise IOError(f"Error writing to file {valid_path}: {e}")

@app.tool()
@log_tool
async def edit_file(args: EditFileArgs) -> str:
    """
    Makes line-based edits to a text file.

    Args:
        args: An EditFileArgs object with path, edits, and dry_run flag.

    Returns:
        A git-style diff string showing the changes made. If not dry_run,
        the file is also modified on disk.

    Raises:
        PermissionError: If path validation fails.
        FileNotFoundError: If the file does not exist.
        ValueError: If an edit's old_text is not found.
        IOError: For other file operation errors.
    """
    valid_path = await validate_path(args.path, is_for_writing=not args.dry_run)
    if not valid_path.is_file():
        raise FileNotFoundError(f"File not found at {valid_path}")

    try:
        with open(valid_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        raise IOError(f"Error reading file {valid_path} for editing: {e}")

    modified_content = _normalize_line_endings(original_content)

    for edit_op in args.edits:
        normalized_old = _normalize_line_endings(edit_op.old_text)
        normalized_new = _normalize_line_endings(edit_op.new_text)
        
        if normalized_old not in modified_content:
            # Attempt a more flexible match if exact match fails, e.g., by ignoring leading/trailing whitespace on lines
            # This part can be made more sophisticated like the JS reference if needed.
            # For now, we stick to exact match after normalization.
            raise ValueError(f"Could not find exact match for edit old_text in {valid_path}:\n---\n{edit_op.old_text}\n---")
        
        # Simple replacement. The JS version has more complex line-by-line logic.
        # Python's replace is global, so if old_text appears multiple times, all are replaced.
        # The spec implies sequential, first-match replacement.
        # To achieve that, we'd need to use re.sub with count=1 or string.find + slicing.
        # For simplicity, using string.replace which replaces all occurrences.
        # If strict first-match-only is needed, this needs refinement.
        # The reference `applyFileEdits` is more complex, handling partial matches and indentation.
        # This is a simplified version.
        
        # Find the first occurrence
        start_index = modified_content.find(normalized_old)
        if start_index == -1:
             raise ValueError(f"Could not find exact match for edit old_text in {valid_path} (after previous edits):\n---\n{edit_op.old_text}\n---")
        
        modified_content = modified_content[:start_index] + normalized_new + modified_content[start_index + len(normalized_old):]


    diff_output = _create_unified_diff(original_content, modified_content, filepath=str(valid_path))

    if not args.dry_run:
        try:
            with open(valid_path, 'w', encoding='utf-8') as f:
                f.write(modified_content) # Write the version with normalized line endings
        except Exception as e:
            raise IOError(f"Error writing modified file {valid_path}: {e}")

    return f"Changes for {valid_path}:\n{diff_output}"

@app.tool()
@log_tool
async def create_directory(args: CreateDirectoryArgs) -> str:
    """
    Creates a new directory. If the directory already exists, it succeeds silently.
    Parent directories will be created if they don't exist.

    Args:
        args: A CreateDirectoryArgs object containing the path.

    Returns:
        A success message string.
    """
    valid_path = await validate_path(args.path, check_existence=False, is_for_writing=True)
    try:
        valid_path.mkdir(parents=True, exist_ok=True)
        return f"Successfully ensured directory exists at {valid_path}"
    except Exception as e:
        raise IOError(f"Error creating directory {valid_path}: {e}")

@app.tool()
@log_tool
async def list_directory(args: ListDirectoryArgs) -> pd.DataFrame:
    """
    Gets a detailed listing of all files and directories in a specified path.

    Args:
        args: A ListDirectoryArgs object containing the path.

    Returns:
        A Pandas DataFrame with columns: 'name', 'type' ('file' or 'directory'), 'path'.
    """
    valid_path = await validate_path(args.path)
    if not valid_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {valid_path}")

    entries = []
    try:
        for entry in valid_path.iterdir():
            # Perform validation for each item being listed to prevent listing contents of symlinked forbidden dirs
            try:
                await validate_path(str(entry.resolve()), check_existence=True) # resolve symlinks before validating
                entry_type = 'directory' if entry.is_dir() else 'file'
                entries.append({'name': entry.name, 'type': entry_type, 'path': str(entry.resolve())})
            except (PermissionError, FileNotFoundError):
                # Skip items that are not accessible or are broken symlinks pointing outside allowed zones
                entries.append({'name': entry.name, 'type': 'inaccessible', 'path': str(entry)})
                
    except Exception as e:
        raise IOError(f"Error listing directory {valid_path}: {e}")
    
    if not entries:
        return pd.DataFrame(columns=['name', 'type', 'path'])
    return pd.DataFrame(entries)


async def _build_tree_recursive(current_path: pathlib.Path, root_validated_path_str: str) -> List[Dict[str, Any]]:
    """Helper to recursively build directory tree, ensuring sub-paths are validated."""
    tree_entries = []
    for entry in current_path.iterdir():
        entry_data: Dict[str, Any] = {'name': entry.name}
        try:
            # Validate each entry before processing
            # We need to ensure that the resolved path of the entry is still within the initial root_validated_path
            # or more broadly, within ALLOWED_DIRECTORIES.
            # The initial validate_path on the root covers the root, but symlinks inside could point out.
            resolved_entry_path = entry.resolve() # Resolves symlinks
            await validate_path(str(resolved_entry_path), check_existence=True) # General validation

            # Specific check: ensure it's still within the tree traversal scope (original validated root)
            # This prevents symlinks from leading the tree traversal outside the initial validated root's hierarchy
            # if the symlink target is allowed but not under the original root.
            # However, the primary security is `validate_path` against `ALLOWED_DIRECTORIES`.
            # If `resolved_entry_path` is valid per `ALLOWED_DIRECTORIES`, it's generally okay to list.

            if entry.is_dir():
                entry_data['type'] = 'directory'
                entry_data['children'] = await _build_tree_recursive(resolved_entry_path, root_validated_path_str)
            else:
                entry_data['type'] = 'file'
            tree_entries.append(entry_data)
        except (PermissionError, FileNotFoundError):
            entry_data['type'] = 'inaccessible' # Mark inaccessible entries
            tree_entries.append(entry_data)
        except Exception: # Catch other potential errors during iteration
            entry_data['type'] = 'error'
            tree_entries.append(entry_data)
            
    return tree_entries

@app.tool()
@log_tool
async def directory_tree(args: DirectoryTreeArgs) -> str:
    """
    Gets a recursive tree view of files and directories as a JSON string.

    Args:
        args: A DirectoryTreeArgs object containing the root path.

    Returns:
        A JSON string representing the directory tree.
        Each entry includes 'name', 'type' ('file'/'directory'/'inaccessible'/'error'),
        and 'children' for directories.
    """
    valid_path = await validate_path(args.path)
    if not valid_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {valid_path}")

    tree_structure = await _build_tree_recursive(valid_path, str(valid_path))
    return json.dumps(tree_structure, indent=2)

@app.tool()
@log_tool
async def move_file(args: MoveFileArgs) -> str:
    """
    Moves or renames a file or directory.

    Args:
        args: A MoveFileArgs object with source and destination paths.

    Returns:
        A success message string.

    Raises:
        PermissionError: If path validation fails for source or destination.
        FileNotFoundError: If the source does not exist.
        FileExistsError: If the destination already exists.
        IOError: For other move operation errors.
    """
    valid_source = await validate_path(args.source, check_existence=True)
    # For destination, it should not exist, but its parent must be valid.
    valid_destination = await validate_path(args.destination, check_existence=False, is_for_writing=True)

    if valid_destination.exists():
        raise FileExistsError(f"Destination already exists: {valid_destination}")
    
    # Ensure destination parent directory exists if moving to a new directory
    valid_destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.move(str(valid_source), str(valid_destination))
        return f"Successfully moved {valid_source} to {valid_destination}"
    except Exception as e:
        raise IOError(f"Error moving {valid_source} to {valid_destination}: {e}")

@app.tool()
@log_tool
async def search_files(args: SearchFilesArgs) -> pd.DataFrame:
    """
    Recursively searches for files and directories matching a pattern.

    Args:
        args: A SearchFilesArgs object with path, pattern, exclude_patterns, and recursive flag.

    Returns:
        A Pandas DataFrame with a single column 'path' containing full paths to matching items.
        Returns an empty DataFrame if no matches are found.
    """
    root_path_str = args.path
    valid_root_path = await validate_path(root_path_str)
    if not valid_root_path.is_dir():
        raise NotADirectoryError(f"Root path for search is not a directory: {valid_root_path}")

    results: List[str] = []
    
    # Normalize exclude patterns to be relative to the root_path for fnmatch
    # This is a simplified approach compared to full .gitignore style matching.
    # For minimatch-like behavior, a more sophisticated library or logic would be needed.
    
    for dirpath_str, dirnames, filenames in os.walk(str(valid_root_path), topdown=True):
        current_dir_path = pathlib.Path(dirpath_str)
        
        # Filter dirnames based on exclude_patterns before further recursion
        # This is a basic exclusion, more complex patterns might need fnmatch on the full path
        if args.exclude_patterns:
            original_dirnames = list(dirnames) # copy before modifying
            dirnames[:] = [d for d in original_dirnames if not any(fnmatch.fnmatch(d, ep) or fnmatch.fnmatch(str(current_dir_path/d), ep) for ep in args.exclude_patterns)]

        all_entries = filenames + dirnames
        
        for entry_name in all_entries:
            full_entry_path = current_dir_path / entry_name
            
            try:
                # Validate each path before adding to results
                # This is important if symlinks are involved or if os.walk somehow bypasses initial validation scope
                await validate_path(str(full_entry_path.resolve()), check_existence=True)

                # Check against exclude patterns (relative and absolute)
                is_excluded = False
                if args.exclude_patterns:
                    relative_path_to_root = str(full_entry_path.relative_to(valid_root_path))
                    if any(fnmatch.fnmatch(entry_name, ep) or \
                           fnmatch.fnmatch(relative_path_to_root, ep) or \
                           fnmatch.fnmatch(str(full_entry_path), ep) for ep in args.exclude_patterns):
                        is_excluded = True
                
                if not is_excluded and fnmatch.fnmatch(entry_name, args.pattern):
                    results.append(str(full_entry_path.resolve())) # Store resolved path
            
            except (PermissionError, FileNotFoundError):
                # Skip inaccessible or invalid paths encountered during search
                continue
            except Exception: # Other errors
                continue # Skip

        if not args.recursive: # If not recursive, stop after the first level
            break
            
    if not results:
        return pd.DataFrame(columns=['path'])
    return pd.DataFrame(results, columns=['path'])

@app.tool()
@log_tool
async def get_file_info(args: GetFileInfoArgs) -> pd.DataFrame:
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
    valid_path = await validate_path(args.path)
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
        
        # Convert the dictionary to a DataFrame with a single row
        return pd.DataFrame([info_dict])

    except Exception as e:
        raise IOError(f"Error getting file info for {valid_path}: {e}")

@app.tool()
@log_tool
async def list_allowed_directories(args: ListAllowedDirectoriesArgs) -> pd.DataFrame:
    """
    Returns the list of directories that this server is allowed to access.

    Args:
        args: ListAllowedDirectoriesArgs (empty).

    Returns:
        A Pandas DataFrame with a single column 'allowed_directory_path'.
    """
    if not ALLOWED_DIRECTORIES:
        return pd.DataFrame(columns=['allowed_directory_path'])
    
    # Ensure all paths in ALLOWED_DIRECTORIES are normalized for consistent output
    normalized_allowed_dirs = [normalize_path(expand_home(d)) for d in ALLOWED_DIRECTORIES]
    
    return pd.DataFrame(normalized_allowed_dirs, columns=['allowed_directory_path'])


# --- Example of how to set ALLOWED_DIRECTORIES (e.g., in server setup) ---
def configure_allowed_directories(directories: List[str]):
    """
    Configures the global ALLOWED_DIRECTORIES list.
    Paths are expanded and normalized.
    This should be called once during server initialization.
    """
    global ALLOWED_DIRECTORIES
    ALLOWED_DIRECTORIES = [normalize_path(expand_home(d)) for d in directories if d]
    # Basic validation: check if configured allowed directories exist
    for d_path_str in ALLOWED_DIRECTORIES:
        d_path = pathlib.Path(d_path_str)
        if not d_path.exists() or not d_path.is_dir():
            # This should ideally be a startup error for the server
            print(f"Warning: Configured allowed directory does not exist or is not a directory: {d_path_str}")
            # Depending on policy, might want to remove it or raise an error.
            # For now, just a warning. A robust server would handle this more strictly.
    if not ALLOWED_DIRECTORIES:
        print("Warning: No allowed directories configured. Filesystem tools will be restricted.")

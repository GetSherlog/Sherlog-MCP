"""Session management middleware for IPython shells."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import dill
from IPython.core.completer import IPCompleter
from IPython.core.interactiveshell import InteractiveShell
from fastmcp.server.middleware import Middleware, MiddlewareContext
from typing import Dict

logger = logging.getLogger("SherlogMCP.SessionMiddleware")

SESSIONS_DIR = Path("/app/data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

SESSION_SHELLS: Dict[str, InteractiveShell] = {}


class SessionMiddleware(Middleware):
    """Middleware that manages IPython shell sessions per MCP session ID."""
    
    def __init__(self, max_sessions: int = 4):
        """Initialize session middleware.
        
        Parameters
        ----------
        max_sessions : int
            Maximum number of concurrent sessions to maintain (default: 4)
        """
        self.max_sessions = max_sessions
        self._lock = asyncio.Lock()
        logger.info(f"SessionMiddleware initialized with max_sessions={max_sessions}")
        logger.debug(f"Sessions directory: {SESSIONS_DIR}")
    
    async def on_request(self, context: MiddlewareContext, call_next):
        """Intercept requests to inject session-specific IPython shell."""
        logger.debug("Processing request through SessionMiddleware")
        
        session_id = None
        if context.fastmcp_context:
            session_id = context.fastmcp_context.session_id
        
        logger.debug(f"Request session_id: {session_id}")
        
        if not session_id:
            logger.debug("No session_id found, using default shell")
            if "default" not in SESSION_SHELLS:
                logger.info("Creating default shell session")
                SESSION_SHELLS["default"] = self._create_shell("default")
            else:
                logger.debug("Using existing default shell session")
        else:
            async with self._lock:
                if session_id in SESSION_SHELLS:
                    logger.info(f"Retrieved existing session: {session_id}")
                else:
                    logger.info(f"Creating new session: {session_id}")
                    SESSION_SHELLS[session_id] = self._create_shell(session_id)
                    logger.info(f"Created new session: {session_id}")
                    
                    logger.debug(f"Current session count: {len(SESSION_SHELLS)}")
                    if len(SESSION_SHELLS) > self.max_sessions:
                        logger.warning(f"Session limit exceeded ({len(SESSION_SHELLS)} > {self.max_sessions}), evicting oldest session")
                        for sid in list(SESSION_SHELLS.keys()):
                            if sid != "default" and len(SESSION_SHELLS) > self.max_sessions:
                                evicted_shell = SESSION_SHELLS.pop(sid)
                                logger.info(f"Evicted session due to limit: {sid}")
                                self._save_session(sid, evicted_shell)
                                break
        
        logger.debug(f"Proceeding with request for session: {session_id or 'default'}")
        result = await call_next(context)
        logger.debug(f"Request completed for session: {session_id or 'default'}")
        
        if session_id and session_id != "default" and session_id in SESSION_SHELLS:
            logger.debug(f"Scheduling async save for session: {session_id}")
            asyncio.create_task(self._save_session_async(session_id, SESSION_SHELLS[session_id]))
        
        return result
    
    def _create_shell(self, session_id: str) -> InteractiveShell:
        """Create a new IPython shell for a session."""
        logger.debug(f"Attempting to create shell for session: {session_id}")
        
        restored_shell = self._restore_session(session_id)
        if restored_shell:
            logger.info(f"Successfully restored shell from disk for session: {session_id}")
            return restored_shell
        
        logger.debug(f"Creating fresh IPython shell for session: {session_id}")
        shell = InteractiveShell.instance()
        shell.reset()

        logger.debug(f"Setting up autoreload for session: {session_id}")
        shell.run_line_magic("load_ext", "autoreload")
        shell.run_line_magic("autoreload", "2")
        shell.Completer = IPCompleter(shell=shell, use_jedi=False)
        
        logger.debug(f"Configuring DataFrame column completion for session: {session_id}")
        def _df_column_matcher(text):
            import re
            import pandas as pd
            try:
                m = re.match(r"(.+?)\['([^\]]*$)", text)
                if not m:
                    return None
                var, stub = m.groups()
                if var in shell.user_ns and isinstance(shell.user_ns[var], pd.DataFrame):
                    cols = shell.user_ns[var].columns.astype(str)
                    return [f"{var}['{c}']" for c in cols if c.startswith(stub)][:200]
            except Exception:
                return None
        
        shell.Completer.custom_matchers.append(_df_column_matcher)
        
        logger.debug(f"Importing default libraries for session: {session_id}")
        shell.run_cell("import pandas as pd\nimport numpy as np\nimport polars as pl")
        
        logger.debug(f"Pushing helper functions for session: {session_id}")
        self._push_helper_functions(shell)
        
        logger.info(f"Successfully created fresh IPython shell for session: {session_id}")
        return shell
    
    def _restore_session(self, session_id: str) -> Optional[InteractiveShell]:
        """Restore a session from disk if it exists."""
        session_file = SESSIONS_DIR / f"{session_id}.pkl"
        
        logger.debug(f"Checking for saved session file: {session_file}")
        if not session_file.exists():
            logger.debug(f"No saved session file found for session: {session_id}")
            return None
        
        logger.info(f"Attempting to restore session from disk: {session_id}")
        try:
            with open(session_file, "rb") as f:
                state = dill.load(f)
            
            logger.debug(f"Successfully loaded session state for: {session_id}")
            shell = InteractiveShell.instance()
            shell.reset()

            logger.debug(f"Setting up restored shell for session: {session_id}")
            shell.run_line_magic("load_ext", "autoreload")
            shell.run_line_magic("autoreload", "2")
            shell.Completer = IPCompleter(shell=shell, use_jedi=False)
     
            user_ns = state.get("user_ns", {})
            logger.debug(f"Restoring {len(user_ns)} variables for session: {session_id}")
            shell.user_ns.update(user_ns)
            
            self._push_helper_functions(shell)
            
            logger.info(f"Successfully restored session from disk: {session_id}")
            return shell
            
        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}")
            logger.debug(f"Removing corrupted session file: {session_file}")
            try:
                session_file.unlink()
                logger.debug(f"Corrupted session file removed: {session_file}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove corrupted session file: {cleanup_error}")
            return None
    
    def _save_session(self, session_id: str, shell: InteractiveShell) -> None:
        """Save session state to disk."""
        if session_id == "default":
            logger.debug("Skipping save for default session")
            return
        
        logger.debug(f"Saving session state for: {session_id}")
        try:
            user_ns = {
                k: v
                for k, v in shell.user_ns.items()
                if not k.startswith("_")
                and k not in {"In", "Out", "exit", "quit", "get_ipython"}
            }
            
            state = {"user_ns": user_ns}
            
            logger.debug(f"Saving {len(user_ns)} variables for session: {session_id}")
            session_file = SESSIONS_DIR / f"{session_id}.pkl"
            with open(session_file, "wb") as f:
                dill.dump(state, f)
            
            logger.info(f"Successfully saved session to disk: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    async def _save_session_async(self, session_id: str, shell: InteractiveShell) -> None:
        """Async wrapper for session saving."""
        logger.debug(f"Starting async save for session: {session_id}")
        try:
            await asyncio.to_thread(self._save_session, session_id, shell)
            logger.debug(f"Completed async save for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in async save for session {session_id}: {e}")
    
    def _push_helper_functions(self, shell: InteractiveShell) -> None:
        """Push helper functions needed by tools into the shell."""
        logger.debug("Loading helper functions for shell")
        
        try:
            from sherlog_mcp.tools.cli_tools import _search_pypi_impl, _query_apt_package_status_impl
            cli_funcs = {
                "_search_pypi_impl": _search_pypi_impl,
                "_query_apt_package_status_impl": _query_apt_package_status_impl,
            }
            logger.debug(f"Loaded {len(cli_funcs)} CLI tool functions")
        except ImportError as e:
            logger.warning(f"Failed to import CLI tools: {e}")
            cli_funcs = {}
        
        try:
            from sherlog_mcp.tools.code_retrieval import (
                _find_method_implementation_impl,
                _find_class_implementation_impl,
                _list_all_methods_impl,
                _list_all_classes_impl,
                _get_codebase_stats_impl,
            )
            code_retrieval_funcs = {
                "_find_method_implementation_impl": _find_method_implementation_impl,
                "_find_class_implementation_impl": _find_class_implementation_impl,
                "_list_all_methods_impl": _list_all_methods_impl,
                "_list_all_classes_impl": _list_all_classes_impl,
                "_get_codebase_stats_impl": _get_codebase_stats_impl,
            }
            logger.debug(f"Loaded {len(code_retrieval_funcs)} code retrieval functions")
        except ImportError as e:
            logger.warning(f"Failed to import code retrieval tools: {e}")
            code_retrieval_funcs = {}
        
        try:
            from sherlog_mcp.tools.external_mcp_tools import convert_to_dataframe
            external_funcs = {"convert_to_dataframe": convert_to_dataframe}
            logger.debug(f"Loaded {len(external_funcs)} external MCP functions")
        except ImportError as e:
            logger.warning(f"Failed to import external MCP tools: {e}")
            external_funcs = {}
        
        all_funcs = {**cli_funcs, **code_retrieval_funcs, **external_funcs}
        if all_funcs:
            shell.push(all_funcs)
            logger.info(f"Successfully pushed {len(all_funcs)} helper functions to shell")
        else:
            logger.warning("No helper functions were loaded")


def get_session_shell(session_id: str) -> Optional[InteractiveShell]:
    """Get shell for a session ID."""
    if not session_id:
        session_id = "default"
    
    logger.debug(f"Retrieving shell for session: {session_id}")
    shell = SESSION_SHELLS.get(session_id)
    
    if shell:
        logger.debug(f"Found shell for session: {session_id}")
    else:
        logger.debug(f"No shell found for session: {session_id}")
    
    return shell
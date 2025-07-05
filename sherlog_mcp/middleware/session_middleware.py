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
    
    async def on_request(self, context: MiddlewareContext, call_next):
        """Intercept requests to inject session-specific IPython shell."""
        session_id = None
        if context.fastmcp_context:
            session_id = context.fastmcp_context.session_id
        
        if not session_id:
            logger.debug("No session_id found, using default shell")
            if "default" not in SESSION_SHELLS:
                SESSION_SHELLS["default"] = self._create_shell("default")
        else:
            async with self._lock:
                if session_id in SESSION_SHELLS:
                    logger.info(f"Retrieved existing session: {session_id}")
                else:
                    SESSION_SHELLS[session_id] = self._create_shell(session_id)
                    logger.info(f"Created new session: {session_id}")
                    
                    if len(SESSION_SHELLS) > self.max_sessions:
                        for sid in list(SESSION_SHELLS.keys()):
                            if sid != "default" and len(SESSION_SHELLS) > self.max_sessions:
                                evicted_shell = SESSION_SHELLS.pop(sid)
                                logger.info(f"Evicted session due to limit: {sid}")
                                self._save_session(sid, evicted_shell)
                                break
        
        result = await call_next(context)
        
        if session_id and session_id != "default" and session_id in SESSION_SHELLS:
            asyncio.create_task(self._save_session_async(session_id, SESSION_SHELLS[session_id]))
        
        return result
    
    def _create_shell(self, session_id: str) -> InteractiveShell:
        """Create a new IPython shell for a session."""
        restored_shell = self._restore_session(session_id)
        if restored_shell:
            return restored_shell
        
        shell = InteractiveShell.instance()
        shell.reset()

        shell.run_line_magic("load_ext", "autoreload")
        shell.run_line_magic("autoreload", "2")
        shell.Completer = IPCompleter(shell=shell, use_jedi=False)
        
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
        
        shell.run_cell("import pandas as pd\nimport numpy as np\nimport polars as pl")
        
        self._push_helper_functions(shell)
        
        logger.debug(f"Created fresh IPython shell for session: {session_id}")
        return shell
    
    def _restore_session(self, session_id: str) -> Optional[InteractiveShell]:
        """Restore a session from disk if it exists."""
        session_file = SESSIONS_DIR / f"{session_id}.pkl"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, "rb") as f:
                state = dill.load(f)
 
            shell = InteractiveShell.instance()
            shell.reset()

            shell.run_line_magic("load_ext", "autoreload")
            shell.run_line_magic("autoreload", "2")
            shell.Completer = IPCompleter(shell=shell, use_jedi=False)
     
            shell.user_ns.update(state.get("user_ns", {}))
            
            # Push helper functions
            self._push_helper_functions(shell)
            
            logger.info(f"Restored session from disk: {session_id}")
            return shell
            
        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}")
            return None
    
    def _save_session(self, session_id: str, shell: InteractiveShell) -> None:
        """Save session state to disk."""
        if session_id == "default":
            return
        
        try:
            state = {
                "user_ns": {
                    k: v
                    for k, v in shell.user_ns.items()
                    if not k.startswith("_")
                    and k not in {"In", "Out", "exit", "quit", "get_ipython"}
                },
            }
            
            session_file = SESSIONS_DIR / f"{session_id}.pkl"
            with open(session_file, "wb") as f:
                dill.dump(state, f)
            
            logger.debug(f"Saved session to disk: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    async def _save_session_async(self, session_id: str, shell: InteractiveShell) -> None:
        """Async wrapper for session saving."""
        await asyncio.to_thread(self._save_session, session_id, shell)
    
    def _push_helper_functions(self, shell: InteractiveShell) -> None:
        """Push helper functions needed by tools into the shell."""
        try:
            from sherlog_mcp.tools.cli_tools import _search_pypi_impl, _query_apt_package_status_impl
            cli_funcs = {
                "_search_pypi_impl": _search_pypi_impl,
                "_query_apt_package_status_impl": _query_apt_package_status_impl,
            }
        except ImportError:
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
        except ImportError:
            code_retrieval_funcs = {}
        
        try:
            from sherlog_mcp.tools.external_mcp_tools import convert_to_dataframe
            external_funcs = {"convert_to_dataframe": convert_to_dataframe}
        except ImportError:
            external_funcs = {}
        
        shell.push({
            **cli_funcs,
            **code_retrieval_funcs,
            **external_funcs
        })
        
        logger.debug(f"Pushed helper functions to shell for session")


def get_session_shell(session_id: str) -> Optional[InteractiveShell]:
    """Get shell for a session ID."""
    if not session_id:
        session_id = "default"
    return SESSION_SHELLS.get(session_id)
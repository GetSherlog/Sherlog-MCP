"""Session management middleware for IPython shells with optimized persistence."""

import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional, Dict, Any, Set
import secrets
import json
import threading
from contextlib import contextmanager

import dill
from IPython.core.completer import IPCompleter
from IPython.core.interactiveshell import InteractiveShell
from fastmcp.server.middleware import Middleware, MiddlewareContext

logger = logging.getLogger("SherlogMCP.SessionMiddleware")

SESSIONS_DIR = Path("/app/data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = SESSIONS_DIR / "sessions.db"

SESSION_SHELLS: Dict[str, InteractiveShell] = {}
SESSION_METADATA: Dict[str, Dict[str, Any]] = {}

# Optimized persistence tracking
DIRTY_SESSIONS: Set[str] = set()  # Sessions that need saving
LAST_SAVE_TIMES: Dict[str, float] = {}  # Last save time per session
SAVE_INTERVAL = 15  # Save dirty sessions every 15 seconds
REQUEST_SAVE_THRESHOLD = 10  # Save after N requests per session
SESSION_REQUEST_COUNTS: Dict[str, int] = {}

# Background persistence worker
_persistence_worker: Optional[asyncio.Task[None]] = None
_shutdown_event = asyncio.Event()

# SQLite connection pool
_db_lock = threading.Lock()


def generate_secure_session_id() -> str:
    """Generate a cryptographically secure session ID."""
    return secrets.token_urlsafe(32)


@contextmanager
def get_db_connection():
    """Thread-safe SQLite connection context manager."""
    with _db_lock:
        conn = sqlite3.connect(DB_FILE, timeout=10.0)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()


def _init_database() -> None:
    """Initialize SQLite database with required tables."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Session metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_initialized REAL,
                    last_accessed REAL,
                    metadata_json TEXT
                )
            """)
            
            # Active sessions registry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_active REAL NOT NULL
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_last_active 
                ON active_sessions(last_active)
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def _save_session_metadata_db(session_id: str, metadata: Dict[str, Any]) -> None:
    """Save session metadata to SQLite database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Serialize only JSON-compatible metadata
            serializable_metadata = {
                k: v for k, v in metadata.items() 
                if isinstance(v, (str, int, float, bool, list, dict))
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO session_metadata 
                (session_id, created_at, last_initialized, last_accessed, metadata_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                metadata.get('created_at'),
                metadata.get('last_initialized'),
                time.time(),  # Update last_accessed
                json.dumps(serializable_metadata)
            ))
            
            conn.commit()
            logger.debug(f"Saved metadata for session: {session_id}")
            
    except Exception as e:
        logger.error(f"Failed to save session metadata for {session_id}: {e}")


def _save_active_session_db(session_id: str) -> None:
    """Register session as active in database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO active_sessions 
                (session_id, created_at, last_active)
                VALUES (?, ?, ?)
            """, (session_id, time.time(), time.time()))
            
            conn.commit()
            logger.debug(f"Registered active session: {session_id}")
            
    except Exception as e:
        logger.error(f"Failed to register active session {session_id}: {e}")


def _remove_active_session_db(session_id: str) -> None:
    """Remove session from active registry in database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            logger.debug(f"Removed active session: {session_id}")
            
    except Exception as e:
        logger.error(f"Failed to remove active session {session_id}: {e}")


def _restore_metadata_db() -> None:
    """Restore session metadata from SQLite database."""
    global SESSION_METADATA
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id, metadata_json FROM session_metadata")
            
            SESSION_METADATA = {}
            for row in cursor.fetchall():
                session_id = row['session_id']
                try:
                    metadata = json.loads(row['metadata_json'])
                    SESSION_METADATA[session_id] = metadata
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for session {session_id}: {e}")
            
            logger.info(f"Restored metadata for {len(SESSION_METADATA)} sessions from database")
            
    except Exception as e:
        logger.error(f"Failed to restore session metadata from database: {e}")
        SESSION_METADATA = {}


def _restore_active_sessions_db() -> list:
    """Restore active session registry from SQLite database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id FROM active_sessions ORDER BY last_active DESC")
            
            active_sessions = [row['session_id'] for row in cursor.fetchall()]
            logger.info(f"Found {len(active_sessions)} active sessions to restore")
            return active_sessions
            
    except Exception as e:
        logger.error(f"Failed to restore active sessions from database: {e}")
        return []


def _mark_session_dirty(session_id: str) -> None:
    """Mark a session as needing persistence."""
    if session_id != "default":
        DIRTY_SESSIONS.add(session_id)
        SESSION_REQUEST_COUNTS[session_id] = SESSION_REQUEST_COUNTS.get(session_id, 0) + 1
        logger.debug(f"Marked session {session_id} as dirty (requests: {SESSION_REQUEST_COUNTS[session_id]})")


async def _persistence_worker_impl() -> None:
    """Background worker for periodic session persistence."""
    logger.info("Starting background persistence worker")
    
    while not _shutdown_event.is_set():
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=SAVE_INTERVAL)
            break  # Shutdown requested
        except asyncio.TimeoutError:
            pass 
        
        if DIRTY_SESSIONS:
            sessions_to_save = DIRTY_SESSIONS.copy()
            DIRTY_SESSIONS.clear()
            
            logger.debug(f"Persistence worker saving {len(sessions_to_save)} dirty sessions")
            
            for session_id in sessions_to_save:
                if session_id in SESSION_SHELLS:
                    try:
                        await _save_session_state_async(session_id, SESSION_SHELLS[session_id])
                        LAST_SAVE_TIMES[session_id] = time.time()
                    except Exception as e:
                        logger.error(f"Failed to save session {session_id} in worker: {e}")
    
    logger.info("Persistence worker stopped")


async def _save_session_state_async(session_id: str, shell: InteractiveShell) -> None:
    """Async wrapper for session state saving."""
    if session_id == "default":
        return
    
    logger.debug(f"Saving session state for: {session_id}")
    try:
        await asyncio.to_thread(_save_session_state, session_id, shell)
        logger.debug(f"Successfully saved session state for: {session_id}")
    except Exception as e:
        logger.error(f"Error saving session state for {session_id}: {e}")


def _save_session_state(session_id: str, shell: InteractiveShell) -> None:
    """Save session state to disk using dill serialization."""
    if session_id == "default":
        return
    
    try:
        # Extract serializable user namespace
        user_ns = {
            k: v
            for k, v in shell.user_ns.items()
            if not k.startswith("_")
            and k not in {"In", "Out", "exit", "quit", "get_ipython"}
        }
        
        state = {"user_ns": user_ns}
        
        session_file = SESSIONS_DIR / f"{session_id}.pkl"
        with open(session_file, "wb") as f:
            dill.dump(state, f)
        
        logger.debug(f"Saved {len(user_ns)} variables for session: {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to save session state {session_id}: {e}")


async def _should_save_session_immediately(session_id: str) -> bool:
    """Check if session should be saved immediately based on request count."""
    if session_id == "default":
        return False
    
    request_count = SESSION_REQUEST_COUNTS.get(session_id, 0)
    return request_count >= REQUEST_SAVE_THRESHOLD


async def start_persistence_worker() -> None:
    """Start the persistence worker during application startup."""
    global _persistence_worker
    
    if _persistence_worker is None or _persistence_worker.done():
        try:
            _persistence_worker = asyncio.create_task(_persistence_worker_impl())
            logger.info("Started background persistence worker during startup")
        except Exception as e:
            logger.error(f"Failed to start persistence worker: {e}")
            raise


async def shutdown_persistence() -> None:
    """Gracefully shutdown persistence and save all dirty sessions."""
    global _persistence_worker
    
    logger.info("Shutting down session persistence...")
    
    # Signal shutdown
    _shutdown_event.set()
    
    # Wait for worker to finish
    if _persistence_worker and not _persistence_worker.done():
        try:
            await asyncio.wait_for(_persistence_worker, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Persistence worker did not stop gracefully")
            _persistence_worker.cancel()
    
    # Save all remaining dirty sessions
    if DIRTY_SESSIONS:
        logger.info(f"Saving {len(DIRTY_SESSIONS)} dirty sessions on shutdown")
        for session_id in list(DIRTY_SESSIONS):
            if session_id in SESSION_SHELLS:
                try:
                    _save_session_state(session_id, SESSION_SHELLS[session_id])
                    logger.debug(f"Saved session {session_id} on shutdown")
                except Exception as e:
                    logger.error(f"Failed to save session {session_id} on shutdown: {e}")
    
    logger.info("Session persistence shutdown complete")


# Initialize database and restore data on module load
_init_database()
_restore_metadata_db()
_pending_session_restores = _restore_active_sessions_db()


class SessionMiddleware(Middleware):
    """Middleware that manages IPython shell sessions with optimized persistence.
    
    Features:
    - SQLite-based metadata and registry storage
    - In-memory session state with periodic snapshots
    - Background persistence worker
    - Dirty flag tracking to minimize I/O
    - Graceful shutdown with state preservation
    """
    
    def __init__(self, max_sessions: int = 4):
        """Initialize session middleware with optimized persistence.
        
        Parameters
        ----------
        max_sessions : int
            Maximum number of concurrent sessions to maintain (default: 4)
        """
        self.max_sessions = max_sessions
        self._lock = asyncio.Lock()
        self._worker_started = False
        logger.info(f"SessionMiddleware initialized with max_sessions={max_sessions}")
        logger.debug(f"Sessions directory: {SESSIONS_DIR}")
        
        # Note: Worker will be started when event loop is available
        
        # Restore sessions that were active
        for session_id in _pending_session_restores:
            if session_id in SESSION_SHELLS:
                continue  # Already restored
                    
            logger.debug(f"Restoring shell for session: {session_id}")
            try:
                restored_shell = self._restore_session(session_id)
                if restored_shell:
                    SESSION_SHELLS[session_id] = restored_shell
                    logger.info(f"Successfully restored shell for session: {session_id}")
                else:
                    # If restoration failed, create a fresh shell
                    logger.info(f"Creating fresh shell for session: {session_id}")
                    SESSION_SHELLS[session_id] = self._create_shell(session_id)
            except Exception as e:
                logger.error(f"Failed to restore session {session_id}: {e}")

    def _start_persistence_worker(self) -> None:
        """Start the background persistence worker."""
        global _persistence_worker
        
        if _persistence_worker is None or _persistence_worker.done():
            try:
                _persistence_worker = asyncio.create_task(_persistence_worker_impl())
                self._worker_started = True
                logger.debug("Started background persistence worker")
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    logger.debug("Event loop not yet available, will start worker on first request")
                    self._worker_started = False
                else:
                    raise

    async def startup(self) -> None:
        """Initialize middleware during app startup when event loop is available."""
        if not self._worker_started:
            self._start_persistence_worker()
            logger.info("Session middleware startup complete")

    async def on_request(self, context: MiddlewareContext, call_next):
        """Process MCP requests with optimized session management."""
        # Start persistence worker if not already started (lazy initialization)
        if not self._worker_started:
            self._start_persistence_worker()
        
        is_initialize = context.method == "initialize"
        
        # Get session ID from context
        session_id = None
        if hasattr(context, 'fastmcp_context') and context.fastmcp_context:
            session_id = getattr(context.fastmcp_context, 'session_id', None)
        
        logger.debug(f"Processing {context.method}, session_id: {session_id}")
        
        # Handle initialize requests
        if is_initialize:
            if session_id and session_id in SESSION_SHELLS:
                # Existing session
                logger.info(f"Reusing existing session: {session_id}")
                SESSION_METADATA[session_id]['last_initialized'] = time.time()
                _save_session_metadata_db(session_id, SESSION_METADATA[session_id])
            else:
                # New session
                session_id = generate_secure_session_id()
                logger.info(f"Generated new session ID: {session_id}")
                
                SESSION_METADATA[session_id] = {
                    'created_at': time.time(),
                    'last_initialized': time.time(),
                    'needs_header': True
                }
                _save_session_metadata_db(session_id, SESSION_METADATA[session_id])
        
        # Ensure shell exists
        if session_id:
            async with self._lock:
                if session_id not in SESSION_SHELLS:
                    logger.info(f"Creating new shell for session: {session_id}")
                    SESSION_SHELLS[session_id] = self._create_shell(session_id)
                    _save_active_session_db(session_id)
                    
                    # Check session limit
                    if len(SESSION_SHELLS) > self.max_sessions:
                        await self._evict_oldest_session()
        else:
            # Default session
            session_id = "default"
            if session_id not in SESSION_SHELLS:
                logger.info("Creating default shell session")
                SESSION_SHELLS[session_id] = self._create_shell(session_id)
        
        # Mark session as accessed
        if session_id != "default":
            _mark_session_dirty(session_id)
        
        # Call next middleware/handler
        result = await call_next(context)
        
        # Handle initialize response
        if is_initialize and session_id != "default" and SESSION_METADATA.get(session_id, {}).get('needs_header'):
            SESSION_METADATA[session_id]['needs_header'] = False
            _save_session_metadata_db(session_id, SESSION_METADATA[session_id])
            
            if hasattr(result, '__dict__'):
                result._mcp_session_id = session_id
            
            logger.info(f"Marked session {session_id} for header injection")
        
        # Check if immediate save is needed based on request frequency
        if session_id and session_id != "default" and session_id in SESSION_SHELLS:
            if await _should_save_session_immediately(session_id):
                logger.debug(f"Immediate save triggered for session {session_id}")
                asyncio.create_task(_save_session_state_async(session_id, SESSION_SHELLS[session_id]))
                SESSION_REQUEST_COUNTS[session_id] = 0  # Reset counter
                DIRTY_SESSIONS.discard(session_id)  # Remove from dirty set
        
        return result
    
    async def _evict_oldest_session(self):
        """Evict the oldest session when limit is exceeded."""
        logger.warning(f"Session limit exceeded ({len(SESSION_SHELLS)} > {self.max_sessions})")
        
        # Find oldest non-default session
        oldest_session = None
        oldest_time = float('inf')
        
        for sid, metadata in SESSION_METADATA.items():
            if sid != "default" and metadata.get('created_at', 0) < oldest_time:
                oldest_time = metadata['created_at']
                oldest_session = sid
        
        if oldest_session and oldest_session in SESSION_SHELLS:
            evicted_shell = SESSION_SHELLS.pop(oldest_session)
            SESSION_METADATA.pop(oldest_session, None)
            logger.info(f"Evicted session due to limit: {oldest_session}")
            
            # Remove from database
            _remove_active_session_db(oldest_session)
            
            # Save evicted session state
            await _save_session_state_async(oldest_session, evicted_shell)
    
    def _create_shell(self, session_id: str) -> InteractiveShell:
        """Create a new IPython shell for a session."""
        logger.debug(f"Creating shell for session: {session_id}")
        
        # Try to restore from disk first
        restored_shell = self._restore_session(session_id)
        if restored_shell:
            logger.info(f"Successfully restored shell from disk for session: {session_id}")
            return restored_shell
        
        logger.debug(f"Creating fresh IPython shell for session: {session_id}")
        shell = InteractiveShell.instance()
        shell.reset()

        # Setup shell configuration
        shell.run_line_magic("load_ext", "autoreload")
        shell.run_line_magic("autoreload", "2")
        shell.Completer = IPCompleter(shell=shell, use_jedi=False)
        
        # Configure DataFrame column completion
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
        
        # Import default libraries
        shell.run_cell("import pandas as pd\nimport numpy as np\nimport polars as pl")
        
        # Push helper functions
        self._push_helper_functions(shell)
        
        logger.info(f"Successfully created fresh IPython shell for session: {session_id}")
        return shell
    
    def _restore_session(self, session_id: str) -> Optional[InteractiveShell]:
        """Restore a session from disk if it exists."""
        session_file = SESSIONS_DIR / f"{session_id}.pkl"
        
        if not session_file.exists():
            logger.debug(f"No saved session file found for session: {session_id}")
            return None
        
        logger.info(f"Attempting to restore session from disk: {session_id}")
        try:
            with open(session_file, "rb") as f:
                state = dill.load(f)
            
            shell = InteractiveShell.instance()
            shell.reset()

            # Setup shell configuration
            shell.run_line_magic("load_ext", "autoreload")
            shell.run_line_magic("autoreload", "2")
            shell.Completer = IPCompleter(shell=shell, use_jedi=False)
     
            # Restore user namespace
            user_ns = state.get("user_ns", {})
            logger.debug(f"Restoring {len(user_ns)} variables for session: {session_id}")
            shell.user_ns.update(user_ns)
            
            self._push_helper_functions(shell)
            
            logger.info(f"Successfully restored session from disk: {session_id}")
            return shell
            
        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}")
            try:
                session_file.unlink()
                logger.debug(f"Removed corrupted session file: {session_file}")
            except Exception:
                pass
            return None
    
    def _push_helper_functions(self, shell: InteractiveShell) -> None:
        """Push helper functions needed by tools into the shell."""
        logger.debug("Loading helper functions for shell")
        
        try:
            from sherlog_mcp.tools.cli_tools import _search_pypi_impl, _query_apt_package_status_impl
            cli_funcs = {
                "_search_pypi_impl": _search_pypi_impl,
                "_query_apt_package_status_impl": _query_apt_package_status_impl,
            }
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
        except ImportError as e:
            logger.warning(f"Failed to import code retrieval tools: {e}")
            code_retrieval_funcs = {}
        
        try:
            from sherlog_mcp.tools.external_mcp_tools import convert_to_dataframe
            external_funcs = {"convert_to_dataframe": convert_to_dataframe}
        except ImportError as e:
            logger.warning(f"Failed to import external MCP tools: {e}")
            external_funcs = {}
        
        all_funcs = {**cli_funcs, **code_retrieval_funcs, **external_funcs}
        if all_funcs:
            shell.push(all_funcs)
            logger.info(f"Successfully pushed {len(all_funcs)} helper functions to shell")


def get_session_shell(session_id: str) -> Optional[InteractiveShell]:
    """Get shell for a session ID."""
    if not session_id:
        session_id = "default"
    
    logger.debug(f"Retrieving shell for session: {session_id}")
    return SESSION_SHELLS.get(session_id)
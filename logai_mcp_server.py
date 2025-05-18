from __future__ import annotations

"""Compatibility wrapper.

This small stub keeps the original *entry point* name so that existing
Docker setups or scripts that call ``python logai_mcp_server.py``
continue to work.  All real functionality now lives in
:pyfile:`logai_mcp/server.py`.
"""

from logai_mcp.server import main

if __name__ == "__main__":
    main() 
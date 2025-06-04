import asyncio
import json
import logging
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

@dataclass
class MCPClientConfig:
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    description: str = ""
    timeout: int = 30
    auto_retry: bool = True

class DynamicToolRegistry:
    """Handles dynamic registration of MCP tools as FastMCP tools"""
    
    def __init__(self, fastmcp_app: FastMCP):
        self.app = fastmcp_app
        self.registered_tools = set()
    
    async def register_mcp_tools(self, mcp_manager):
        """Register all tools from active MCP clients as individual FastMCP tools"""
        
        for client_name, session in mcp_manager.active_sessions.items():
            if not session:
                continue
                
            try:
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    await self._register_individual_tool(client_name, tool, session)
                    
                logger.info(f"Registered {len(tools_result.tools)} tools from {client_name}")
            except Exception as e:
                logger.warning(f"Failed to register tools for {client_name}: {e}")
    
    async def _register_individual_tool(self, client_name: str, tool, session):
        """Register a single MCP tool as a FastMCP tool"""
        
        # Create unique tool name: clientname_toolname
        fastmcp_tool_name = f"{client_name}_{tool.name}"
        
        # Skip if already registered
        if fastmcp_tool_name in self.registered_tools:
            return
        
        # Extract parameter info from tool schema
        params = self._extract_parameters(tool)
        
        # Create the dynamic function with proper closure
        def make_tool_function(client_name, tool_name, tool_description, session_ref):
            async def dynamic_tool_func(**kwargs):
                try:
                    # Get the current session (in case of reconnects)
                    current_session = session_ref() if callable(session_ref) else session_ref
                    if not current_session:
                        return {"error": f"MCP client '{client_name}' is not connected"}
                    
                    result = await current_session.call_tool(tool_name, kwargs)
                    return result.content
                except Exception as e:
                    logger.error(f"Error calling {client_name}.{tool_name}: {e}")
                    return {"error": f"Error calling {client_name}.{tool_name}: {str(e)}"}
            
            # Set proper metadata
            dynamic_tool_func.__name__ = fastmcp_tool_name
            dynamic_tool_func.__doc__ = f"[MCP:{client_name}] {tool_description}"
            
            # Create proper signature for FastMCP type checking
            if params:
                sig = self._create_signature(params)
                dynamic_tool_func.__signature__ = sig
                dynamic_tool_func.__annotations__ = self._create_annotations(params)
            
            return dynamic_tool_func
        
        # Create a session getter to handle reconnections
        session_getter = lambda: session
        
        # Create and register the tool function
        tool_func = make_tool_function(client_name, tool.name, tool.description, session_getter)
        
        # Register with FastMCP
        self.app.tool()(tool_func)
        self.registered_tools.add(fastmcp_tool_name)
        
        logger.debug(f"Registered MCP tool: {fastmcp_tool_name}")
    
    def _extract_parameters(self, tool):
        """Extract parameter definitions from MCP tool schema"""
        if hasattr(tool, 'inputSchema') and tool.inputSchema:
            properties = tool.inputSchema.get('properties', {})
            required = tool.inputSchema.get('required', [])
            
            params = {}
            for param_name, param_def in properties.items():
                params[param_name] = {
                    'type': param_def.get('type', 'string'),
                    'description': param_def.get('description', ''),
                    'required': param_name in required,
                    'default': param_def.get('default')
                }
            return params
        return {}
    
    def _create_signature(self, params):
        """Create proper function signature for FastMCP type checking"""
        parameters = []
        for param_name, param_info in params.items():
            param_type = self._json_type_to_python(param_info['type'])
            
            if param_info['required']:
                param = inspect.Parameter(
                    param_name, 
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=param_type
                )
            else:
                default = param_info.get('default', None)
                param = inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                    annotation=param_type,
                    default=default
                )
            parameters.append(param)
        
        return inspect.Signature(parameters)
    
    def _create_annotations(self, params):
        """Create type annotations dict"""
        annotations = {}
        for param_name, param_info in params.items():
            annotations[param_name] = self._json_type_to_python(param_info['type'])
        annotations['return'] = Any
        return annotations
    
    def _json_type_to_python(self, json_type):
        """Convert JSON schema types to Python types"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        return type_mapping.get(json_type, str)

class MCPManager:
    """Manages MCP client connections and dynamic tool registration"""
    
    def __init__(self, fastmcp_app: FastMCP, config_path: str = "mcp_config.json"):
        self.app = fastmcp_app
        self.config_path = Path(config_path)
        self.active_sessions: Dict[str, ClientSession] = {}
        self.connection_params: Dict[str, MCPClientConfig] = {}
        self.tool_registry = DynamicToolRegistry(fastmcp_app)
        self._lock = asyncio.Lock()
        
        # Ensure config file exists
        if not self.config_path.exists():
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default MCP configuration file"""
        default_config = {
            "mcp_clients": {
                "example": {
                    "command": "python",
                    "args": ["examples/example_server.py"],
                    "env": {},
                    "description": "Example MCP server - remove when adding real servers",
                    "timeout": 30,
                    "auto_retry": True
                }
            }
        }
        self.config_path.write_text(json.dumps(default_config, indent=2))
        logger.info(f"Created default MCP config at {self.config_path}")
    
    async def load_config(self) -> Dict[str, Any]:
        """Load MCP client configuration from file"""
        try:
            config_text = self.config_path.read_text()
            return json.loads(config_text)
        except Exception as e:
            logger.error(f"Error loading MCP config: {e}")
            return {"mcp_clients": {}}
    
    async def save_config(self, config: Dict[str, Any]):
        """Save MCP client configuration to file"""
        try:
            self.config_path.write_text(json.dumps(config, indent=2))
            logger.info("Saved MCP configuration")
        except Exception as e:
            logger.error(f"Error saving MCP config: {e}")
    
    async def initialize_from_config(self):
        """Initialize all MCP clients from configuration file"""
        config = await self.load_config()
        
        for name, client_config in config.get("mcp_clients", {}).items():
            try:
                mcp_config = MCPClientConfig(
                    name=name,
                    command=client_config["command"],
                    args=client_config.get("args", []),
                    env=client_config.get("env", {}),
                    description=client_config.get("description", ""),
                    timeout=client_config.get("timeout", 30),
                    auto_retry=client_config.get("auto_retry", True)
                )
                
                await self._connect_client(mcp_config)
                
            except Exception as e:
                logger.warning(f"Failed to initialize MCP client {name}: {e}")
        
        # Register all tools from successfully connected clients
        await self.tool_registry.register_mcp_tools(self)
        logger.info(f"Initialized {len(self.active_sessions)} MCP clients")
    
    async def add_client(self, config: MCPClientConfig) -> bool:
        """Add and connect to new MCP client"""
        async with self._lock:
            try:
                # Save to config file
                file_config = await self.load_config()
                file_config["mcp_clients"][config.name] = {
                    "command": config.command,
                    "args": config.args,
                    "env": config.env,
                    "description": config.description,
                    "timeout": config.timeout,
                    "auto_retry": config.auto_retry
                }
                await self.save_config(file_config)
                
                # Connect to client
                success = await self._connect_client(config)
                if success:
                    # Register tools from this specific client
                    await self.tool_registry.register_mcp_tools(self)
                    logger.info(f"Successfully added MCP client: {config.name}")
                    return True
                else:
                    logger.warning(f"Failed to connect to MCP client: {config.name}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error adding MCP client {config.name}: {e}")
                return False
    
    async def _connect_client(self, config: MCPClientConfig) -> bool:
        """Connect to a single MCP client"""
        try:
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env
            )
            
            # Use timeout for connection
            async with asyncio.timeout(config.timeout):
                transport = await stdio_client(server_params).__aenter__()
                read, write = transport
                session = ClientSession(read, write)
                await session.initialize()
                
                self.active_sessions[config.name] = session
                self.connection_params[config.name] = config
                
                logger.info(f"Connected to MCP client: {config.name}")
                return True
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout connecting to MCP client {config.name}")
            return False
        except FileNotFoundError:
            logger.warning(f"Command not found for MCP client {config.name}: {config.command}")
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to MCP client {config.name}: {e}")
            return False
    
    async def remove_client(self, name: str) -> bool:
        """Remove MCP client and close connection"""
        async with self._lock:
            try:
                # Close session if active
                if name in self.active_sessions:
                    session = self.active_sessions[name]
                    # ClientSession doesn't have close(), just remove from tracking
                    del self.active_sessions[name]
                
                # Remove from connection params
                if name in self.connection_params:
                    del self.connection_params[name]
                
                # Remove from config file
                config = await self.load_config()
                if name in config.get("mcp_clients", {}):
                    del config["mcp_clients"][name]
                    await self.save_config(config)
                
                logger.info(f"Removed MCP client: {name}")
                return True
                
            except Exception as e:
                logger.error(f"Error removing MCP client {name}: {e}")
                return False
    
    async def reload_all_clients(self):
        """Reload all MCP clients from configuration"""
        async with self._lock:
            # Close existing connections by clearing the sessions
            # The transport cleanup will happen automatically
            self.active_sessions.clear()
            self.connection_params.clear()
            
            # Reload from config
            await self.initialize_from_config()
    
    def get_session(self, client_name: str) -> Optional[ClientSession]:
        """Get active session for MCP client"""
        return self.active_sessions.get(client_name)
    
    def list_clients(self) -> Dict[str, Dict[str, Any]]:
        """List all configured MCP clients and their status"""
        result = {}
        for name, config in self.connection_params.items():
            result[name] = {
                "description": config.description,
                "command": config.command,
                "args": config.args,
                "connected": name in self.active_sessions,
                "timeout": config.timeout,
                "auto_retry": config.auto_retry
            }
        return result
    
    async def get_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available tools from all connected MCP clients"""
        all_tools = {}
        
        for client_name, session in self.active_sessions.items():
            if not session:
                continue
                
            try:
                tools_result = await session.list_tools()
                all_tools[client_name] = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        "fastmcp_name": f"{client_name}_{tool.name}"
                    }
                    for tool in tools_result.tools
                ]
            except Exception as e:
                logger.warning(f"Error getting tools from {client_name}: {e}")
                all_tools[client_name] = []
        
        return all_tools

# Global MCP manager instance (will be initialized when app is created)
mcp_manager: Optional[MCPManager] = None

def get_mcp_manager() -> MCPManager:
    """Get the global MCP manager instance"""
    if mcp_manager is None:
        raise RuntimeError("MCP manager not initialized")
    return mcp_manager

def initialize_mcp_manager(fastmcp_app: FastMCP, config_path: str = "mcp_config.json") -> MCPManager:
    """Initialize the global MCP manager"""
    global mcp_manager
    mcp_manager = MCPManager(fastmcp_app, config_path)
    return mcp_manager 
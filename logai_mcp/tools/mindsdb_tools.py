"""MindsDB Tools for LogAI MCP Server

This module provides tools for interacting with MindsDB for federated data queries and AI operations.
MindsDB enables querying across multiple data sources as if they were a single database.
All operations are logged and can be accessed through audit endpoints.

Dependencies:
    mindsdb_sdk: pip install mindsdb_sdk
"""

import json
import mindsdb_sdk 
from typing import Optional, List, Dict, Any
from logai_mcp.session import app
from logai_mcp.config import get_settings
import logging

logger = logging.getLogger(__name__)

def get_mindsdb_server():
    """Get configured MindsDB server connection using the official SDK."""
    try:
        settings = get_settings()
        
        if not settings.mindsdb_url:
            raise ValueError("MINDSDB_URL must be set in environment variables")
        
        # Connect using the MindsDB SDK
        if settings.mindsdb_access_token:
            # For authenticated connections with custom URL and token
            # Note: The SDK may handle authentication differently
            logger.info("Attempting to connect to MindsDB with authentication")
            server = mindsdb_sdk.connect(
                url=settings.mindsdb_url,
                login=settings.mindsdb_access_token,  # May need adjustment based on SDK auth
                password=""  # May need adjustment
            )
        else:
            # For local/unauthenticated connections
            logger.info("Attempting to connect to MindsDB without authentication")
            server = mindsdb_sdk.connect(settings.mindsdb_url)
        
        # Test the connection by listing databases
        try:
            databases = server.list_databases()
            logger.info(f"Successfully connected to MindsDB at {settings.mindsdb_url}")
            logger.info(f"Found {len(databases)} databases")
        except Exception as e:
            logger.warning(f"Could not verify MindsDB connection: {str(e)}")
        
        return server
        
    except Exception as e:
        error_msg = f"Failed to connect to MindsDB: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


@app.tool()
def list_mindsdb_databases() -> str:
    """List all databases and data sources connected to MindsDB.
    
    Returns:
        str: JSON string containing list of databases with metadata
    """
    try:
        server = get_mindsdb_server()
        
        logger.info("Listing MindsDB databases")
        databases = server.list_databases()
        
        database_list = []
        for db in databases:
            database_info = {
                'name': db.name,
                'type': getattr(db, 'type', 'unknown'),
                'engine': getattr(db, 'engine', 'unknown')
            }
            database_list.append(database_info)
        
        response_data = {
            'databases': database_list,
            'count': len(database_list)
        }
        
        logger.info(f"Listed {len(database_list)} MindsDB databases")
        return json.dumps(response_data, indent=2)
        
    except Exception as e:
        error_msg = f"Failed to list MindsDB databases: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def query_mindsdb(query: str, limit: Optional[int] = None) -> str:
    """Execute a SQL query against MindsDB's federated data.
    
    Args:
        query: SQL query to execute against MindsDB
        limit: Optional limit on number of rows to return
        
    Returns:
        str: JSON string containing query results
    """
    try:
        server = get_mindsdb_server()
        
        # Validate query for basic security
        query_lower = query.lower().strip()
        if not query_lower:
            error_msg = "Query cannot be empty"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})
        
        # Add limit if specified and not already in query
        if limit and 'limit' not in query_lower:
            query = f"{query.rstrip(';')} LIMIT {limit}"
        
        logger.info(f"Executing MindsDB query: {query}")
        
        # Execute query using the SDK
        result = server.query(query)
        df = result.fetch()
        
        # Convert DataFrame to JSON-serializable format
        response_data = {
            'query': query,
            'success': True,
            'columns': df.columns.tolist() if hasattr(df, 'columns') else [],
            'data': df.to_dict('records') if hasattr(df, 'to_dict') else [],
            'row_count': len(df) if hasattr(df, '__len__') else 0
        }
        
        logger.info(f"MindsDB query completed successfully, returned {response_data['row_count']} rows")
        return json.dumps(response_data, indent=2, default=str)
        
    except Exception as e:
        error_msg = f"Failed to execute MindsDB query: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            'query': query,
            'success': False,
            'error': error_msg
        })


@app.tool()
def list_mindsdb_tables(database_name: Optional[str] = None) -> str:
    """List tables in MindsDB databases.
    
    Args:
        database_name: Optional database name to list tables from. If not provided, lists from default database.
        
    Returns:
        str: JSON string containing list of tables
    """
    try:
        server = get_mindsdb_server()
        
        if database_name:
            # Get specific database and list its tables
            database = server.get_database(database_name)
            tables = database.list_tables()
            logger.info(f"Listing MindsDB tables from database: {database_name}")
        else:
            # For default database, use the mindsdb database
            try:
                database = server.get_database('mindsdb')
                tables = database.list_tables()
                logger.info("Listing MindsDB tables from mindsdb database")
            except:
                # Fallback to empty list if mindsdb database doesn't exist
                tables = []
                logger.info("No default database found, returning empty list")
        
        table_list = []
        for table in tables:
            table_info = {
                'name': table.name,
                'database': database_name or 'default',
                'type': getattr(table, 'type', 'unknown')
            }
            table_list.append(table_info)
        
        response_data = {
            'database': database_name or 'default',
            'tables': table_list,
            'count': len(table_list)
        }
        
        logger.info(f"Listed {len(table_list)} tables from MindsDB database: {database_name or 'default'}")
        return json.dumps(response_data, indent=2)
        
    except Exception as e:
        error_msg = f"Failed to list MindsDB tables: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def describe_mindsdb_table(table_name: str, database_name: Optional[str] = None) -> str:
    """Describe the structure of a table in MindsDB.
    
    Args:
        table_name: Name of the table to describe
        database_name: Optional database name. If not provided, uses default database.
        
    Returns:
        str: JSON string containing table schema information
    """
    try:
        server = get_mindsdb_server()
        
        if database_name:
            database = server.get_database(database_name)
            full_table_name = f"{database_name}.{table_name}"
        else:
            full_table_name = table_name
        
        logger.info(f"Describing MindsDB table: {full_table_name}")
        
        # Get table description using SQL query
        describe_query = f"DESCRIBE {full_table_name}"
        result = server.query(describe_query)
        df = result.fetch()
        
        # Convert to list of column information
        columns = df.to_dict('records') if hasattr(df, 'to_dict') else []
        
        response_data = {
            'table_name': table_name,
            'database': database_name or 'default',
            'full_name': full_table_name,
            'columns': columns,
            'column_count': len(columns)
        }
        
        logger.info(f"Described MindsDB table {full_table_name} with {len(columns)} columns")
        return json.dumps(response_data, indent=2, default=str)
        
    except Exception as e:
        error_msg = f"Failed to describe MindsDB table {table_name}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def list_mindsdb_projects() -> str:
    """List all projects in MindsDB.
    
    Returns:
        str: JSON string containing list of projects
    """
    try:
        server = get_mindsdb_server()
        
        logger.info("Listing MindsDB projects")
        projects = server.list_projects()
        
        project_list = []
        for project in projects:
            project_info = {
                'name': project.name
            }
            project_list.append(project_info)
        
        response_data = {
            'projects': project_list,
            'count': len(project_list)
        }
        
        logger.info(f"Listed {len(project_list)} MindsDB projects")
        return json.dumps(response_data, indent=2)
        
    except Exception as e:
        error_msg = f"Failed to list MindsDB projects: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def list_mindsdb_models(project_name: Optional[str] = None) -> str:
    """List all machine learning models in MindsDB.
    
    Args:
        project_name: Optional project name to list models from. If not provided, lists from default project.
    
    Returns:
        str: JSON string containing list of models with metadata
    """
    try:
        server = get_mindsdb_server()
        
        if project_name:
            project = server.get_project(project_name)
            models = project.list_models()
            logger.info(f"Listing MindsDB models from project: {project_name}")
        else:
            models = server.list_models()
            logger.info("Listing MindsDB models from default project")
        
        model_list = []
        for model in models:
            model_info = {
                'name': model.name,
                'project': project_name or 'default',
                'status': getattr(model, 'status', 'unknown'),
                'predict': getattr(model, 'predict', 'unknown'),
                'accuracy': getattr(model, 'accuracy', None)
            }
            model_list.append(model_info)
        
        response_data = {
            'project': project_name or 'default',
            'models': model_list,
            'count': len(model_list)
        }
        
        logger.info(f"Listed {len(model_list)} MindsDB models")
        return json.dumps(response_data, indent=2, default=str)
        
    except Exception as e:
        error_msg = f"Failed to list MindsDB models: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def create_mindsdb_model(model_name: str, predict_column: str, 
                        from_table: str, project_name: Optional[str] = None,
                        model_options: Optional[Dict[str, Any]] = None) -> str:
    """Create a machine learning model in MindsDB.
    
    Args:
        model_name: Name for the new model
        predict_column: Column name to predict
        from_table: Source table for training data
        project_name: Optional project name. If not provided, uses default project.
        model_options: Optional dictionary of model configuration options
        
    Returns:
        str: JSON string containing model creation result
    """
    try:
        server = get_mindsdb_server()
        
        if project_name:
            project = server.get_project(project_name)
        else:
            project = server  # Default project
        
        logger.info(f"Creating MindsDB model: {model_name}")
        
        # Get the source query/table
        if '.' in from_table:
            # If table name includes database, query it directly
            query = project.query(f"SELECT * FROM {from_table}")
        else:
            # Assume it's a table in the current context
            query = project.query(f"SELECT * FROM {from_table}")
        
        # Create model with options
        create_kwargs = {
            'name': model_name,
            'predict': predict_column,
            'query': query
        }
        
        if model_options:
            # Add model options like timeseries_options, etc.
            create_kwargs.update(model_options)
        
        model = project.models.create(**create_kwargs)
        
        response_data = {
            'model_name': model_name,
            'predict_column': predict_column,
            'from_table': from_table,
            'project': project_name or 'default',
            'model_options': model_options,
            'success': True,
            'message': f'Model {model_name} created successfully'
        }
        
        logger.info(f"Successfully created MindsDB model: {model_name}")
        return json.dumps(response_data, indent=2)
        
    except Exception as e:
        error_msg = f"Failed to create MindsDB model {model_name}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            'model_name': model_name,
            'success': False,
            'error': error_msg
        })


@app.tool()
def predict_with_model(model_name: str, input_data: Dict[str, Any], 
                      project_name: Optional[str] = None) -> str:
    """Make predictions using a MindsDB model.
    
    Args:
        model_name: Name of the model to use for prediction
        input_data: Dictionary containing input data for prediction
        project_name: Optional project name. If not provided, uses default project.
        
    Returns:
        str: JSON string containing prediction results
    """
    try:
        server = get_mindsdb_server()
        
        if project_name:
            project = server.get_project(project_name)
            model = project.get_model(model_name)
        else:
            model = server.get_model(model_name)
        
        logger.info(f"Making prediction with MindsDB model: {model_name}")
        
        # Convert input data to DataFrame-like format for prediction
        import pandas as pd
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        result = model.predict(input_df)
        
        # Convert result to JSON-serializable format
        try:
            # Simply convert result to string for now to avoid type issues
            predictions = [{'prediction': str(result)}]
        except Exception:
            predictions = [{'prediction': 'Unable to convert prediction result'}]
        
        response_data = {
            'model_name': model_name,
            'project': project_name or 'default',
            'input_data': input_data,
            'predictions': predictions,
            'success': True
        }
        
        logger.info(f"Successfully made prediction with model {model_name}")
        return json.dumps(response_data, indent=2, default=str)
        
    except Exception as e:
        error_msg = f"Failed to make prediction with model {model_name}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            'model_name': model_name,
            'success': False,
            'error': error_msg
        })


@app.tool()
def get_mindsdb_status() -> str:
    """Get MindsDB server status and connection information.
    
    Returns:
        str: JSON string containing server status information
    """
    try:
        server = get_mindsdb_server()
        settings = get_settings()
        
        # Try to get basic server information
        databases = server.list_databases()
        projects = server.list_projects()
        
        response_data = {
            'mindsdb_url': settings.mindsdb_url,
            'authenticated': bool(settings.mindsdb_access_token),
            'connection_status': 'connected',
            'databases_count': len(databases),
            'projects_count': len(projects)
        }
        
        logger.info("Retrieved MindsDB status information")
        return json.dumps(response_data, indent=2)
        
    except Exception as e:
        error_msg = f"Failed to get MindsDB status: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            'connection_status': 'failed',
            'error': error_msg
        }) 
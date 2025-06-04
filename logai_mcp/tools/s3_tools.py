"""AWS S3 Tools for LogAI MCP Server

This module provides tools for interacting with Amazon S3 buckets and objects.
All operations are logged and can be accessed through audit endpoints.
"""

import os
import json
import boto3
from typing import Optional
from botocore.exceptions import ClientError, NoCredentialsError
from logai_mcp.session import app
from logai_mcp.config import get_settings
import logging

logger = logging.getLogger(__name__)

def get_s3_client():
    """Get configured S3 client with credentials from centralized config or AWS credential chain."""
    try:
        settings = get_settings()
        
        # Create session with credentials from settings if available
        session_kwargs = {}
        
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = settings.aws_access_key_id
            session_kwargs['aws_secret_access_key'] = settings.aws_secret_access_key
            
            if settings.aws_session_token:
                session_kwargs['aws_session_token'] = settings.aws_session_token
                
            logger.info("Using AWS credentials from configuration settings")
        else:
            # Fall back to default credential chain (environment variables, ~/.aws/credentials, IAM roles, etc.)
            logger.info("Using AWS default credential chain")
        
        session = boto3.Session(**session_kwargs)
        s3_client = session.client('s3', region_name=settings.aws_region)
        
        # Test the connection by listing buckets (this will fail if no credentials)
        s3_client.list_buckets()
        
        logger.info(f"Successfully initialized S3 client for region: {settings.aws_region}")
        return s3_client
        
    except NoCredentialsError:
        error_msg = """
        AWS credentials not found. Please configure credentials using one of these methods:
        
        1. Environment variables:
           export AWS_ACCESS_KEY_ID=your_access_key
           export AWS_SECRET_ACCESS_KEY=your_secret_key
           export AWS_REGION=us-east-1
        
        2. Configuration in .env file:
           AWS_ACCESS_KEY_ID=your_access_key
           AWS_SECRET_ACCESS_KEY=your_secret_key
           AWS_REGION=us-east-1
        
        3. AWS CLI: aws configure
        
        4. AWS credentials file (~/.aws/credentials)
        
        5. IAM roles (if running on EC2)
        """
        logger.error(error_msg)
        raise Exception(f"AWS credentials not configured: {error_msg}")
        
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {str(e)}")
        raise Exception(f"Failed to initialize S3 client: {str(e)}")


@app.tool()
def list_s3_buckets() -> str:
    """List all S3 buckets in the AWS account.
    
    Returns:
        str: JSON string containing list of bucket information including names and creation dates.
    """
    try:
        s3_client = get_s3_client()
        response = s3_client.list_buckets()
        
        buckets = []
        for bucket in response.get('Buckets', []):
            buckets.append({
                'name': bucket['Name'],
                'creation_date': bucket['CreationDate'].isoformat() if bucket.get('CreationDate') else None
            })
        
        result = {
            'buckets': buckets,
            'count': len(buckets)
        }
        
        logger.info(f"Listed {len(buckets)} S3 buckets")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Failed to list S3 buckets: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def create_s3_bucket(bucket_name: str, region: Optional[str] = None) -> str:
    """Create a new S3 bucket.
    
    Args:
        bucket_name: Name of the bucket to create (must be globally unique)
        region: AWS region for the bucket (defaults to configured region)
    
    Returns:
        str: JSON string with creation result
    """
    try:
        s3_client = get_s3_client()
        settings = get_settings()
        
        if not region:
            region = settings.aws_region
        
        # Create bucket with appropriate configuration
        if region == 'us-east-1':
            # us-east-1 doesn't need LocationConstraint
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        
        result = {
            'success': True,
            'bucket_name': bucket_name,
            'region': region,
            'message': f'Bucket {bucket_name} created successfully in {region}'
        }
        
        logger.info(f"Created S3 bucket: {bucket_name} in region: {region}")
        return json.dumps(result, indent=2)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'BucketAlreadyExists':
            error_msg = f"Bucket {bucket_name} already exists and is owned by another account"
        elif error_code == 'BucketAlreadyOwnedByYou':
            error_msg = f"Bucket {bucket_name} already exists and is owned by you"
        else:
            error_msg = f"Failed to create bucket {bucket_name}: {str(e)}"
            
        logger.error(error_msg)
        return json.dumps({'error': error_msg})
        
    except Exception as e:
        error_msg = f"Failed to create bucket {bucket_name}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def delete_s3_bucket(bucket_name: str, force: bool = False) -> str:
    """Delete an S3 bucket.
    
    Args:
        bucket_name: Name of the bucket to delete
        force: If True, delete all objects in the bucket first (use with caution!)
    
    Returns:
        str: JSON string with deletion result
    """
    try:
        s3_client = get_s3_client()
        
        if force:
            # First delete all objects in the bucket
            logger.warning(f"Force deleting all objects in bucket: {bucket_name}")
            
            # List and delete all objects
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name):
                if 'Contents' in page:
                    objects = [{'Key': obj['Key']} for obj in page['Contents']]
                    s3_client.delete_objects(
                        Bucket=bucket_name,
                        Delete={'Objects': objects}
                    )
            
            # List and delete all object versions (for versioned buckets)
            try:
                version_paginator = s3_client.get_paginator('list_object_versions')
                for page in version_paginator.paginate(Bucket=bucket_name):
                    if 'Versions' in page:
                        versions = [{'Key': obj['Key'], 'VersionId': obj['VersionId']} 
                                   for obj in page['Versions']]
                        s3_client.delete_objects(
                            Bucket=bucket_name,
                            Delete={'Objects': versions}
                        )
                    if 'DeleteMarkers' in page:
                        delete_markers = [{'Key': obj['Key'], 'VersionId': obj['VersionId']} 
                                        for obj in page['DeleteMarkers']]
                        s3_client.delete_objects(
                            Bucket=bucket_name,
                            Delete={'Objects': delete_markers}
                        )
            except:
                pass  # Bucket might not have versioning enabled
        
        # Delete the bucket
        s3_client.delete_bucket(Bucket=bucket_name)
        
        result = {
            'success': True,
            'bucket_name': bucket_name,
            'message': f'Bucket {bucket_name} deleted successfully'
        }
        
        logger.info(f"Deleted S3 bucket: {bucket_name}")
        return json.dumps(result, indent=2)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'BucketNotEmpty':
            error_msg = f"Bucket {bucket_name} is not empty. Use force=True to delete all objects first"
        elif error_code == 'NoSuchBucket':
            error_msg = f"Bucket {bucket_name} does not exist"
        else:
            error_msg = f"Failed to delete bucket {bucket_name}: {str(e)}"
            
        logger.error(error_msg)
        return json.dumps({'error': error_msg})
        
    except Exception as e:
        error_msg = f"Failed to delete bucket {bucket_name}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def list_s3_objects(bucket_name: str, prefix: str = "", max_keys: int = 100) -> str:
    """List objects in an S3 bucket.
    
    Args:
        bucket_name: Name of the S3 bucket
        prefix: Filter objects by prefix (optional)
        max_keys: Maximum number of objects to return (default 100, max 1000)
    
    Returns:
        str: JSON string containing list of objects with metadata
    """
    try:
        s3_client = get_s3_client()
        
        # Limit max_keys to prevent overwhelming responses
        max_keys = min(max_keys, 1000)
        
        params = {
            'Bucket': bucket_name,
            'MaxKeys': max_keys
        }
        
        if prefix:
            params['Prefix'] = prefix
        
        response = s3_client.list_objects_v2(**params)
        
        objects = []
        for obj in response.get('Contents', []):
            objects.append({
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'].isoformat() if obj.get('LastModified') else None,
                'etag': obj.get('ETag', '').strip('"'),
                'storage_class': obj.get('StorageClass', 'STANDARD')
            })
        
        result = {
            'bucket': bucket_name,
            'prefix': prefix,
            'objects': objects,
            'count': len(objects),
            'is_truncated': response.get('IsTruncated', False),
            'total_size_bytes': sum(obj['size'] for obj in objects)
        }
        
        logger.info(f"Listed {len(objects)} objects in S3 bucket: {bucket_name}")
        return json.dumps(result, indent=2)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            error_msg = f"Bucket {bucket_name} does not exist"
        else:
            error_msg = f"Failed to list objects in bucket {bucket_name}: {str(e)}"
            
        logger.error(error_msg)
        return json.dumps({'error': error_msg})
        
    except Exception as e:
        error_msg = f"Failed to list objects in bucket {bucket_name}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def upload_s3_object(bucket_name: str, key: str, file_path: str, 
                     content_type: Optional[str] = None) -> str:
    """Upload a file to S3.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key (path/filename in S3)
        file_path: Local file path to upload
        content_type: MIME type of the file (auto-detected if not provided)
    
    Returns:
        str: JSON string with upload result
    """
    try:
        import mimetypes
        
        s3_client = get_s3_client()
        
        # Check if local file exists
        if not os.path.exists(file_path):
            error_msg = f"Local file does not exist: {file_path}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})
        
        # Auto-detect content type if not provided
        if not content_type:
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = 'application/octet-stream'
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Upload file
        extra_args = {'ContentType': content_type}
        s3_client.upload_file(file_path, bucket_name, key, ExtraArgs=extra_args)
        
        result = {
            'success': True,
            'bucket': bucket_name,
            'key': key,
            'file_path': file_path,
            'content_type': content_type,
            'size_bytes': file_size,
            's3_url': f's3://{bucket_name}/{key}',
            'message': f'File uploaded successfully to s3://{bucket_name}/{key}'
        }
        
        logger.info(f"Uploaded file {file_path} to s3://{bucket_name}/{key}")
        return json.dumps(result, indent=2)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            error_msg = f"Bucket {bucket_name} does not exist"
        else:
            error_msg = f"Failed to upload {file_path} to s3://{bucket_name}/{key}: {str(e)}"
            
        logger.error(error_msg)
        return json.dumps({'error': error_msg})
        
    except Exception as e:
        error_msg = f"Failed to upload {file_path} to s3://{bucket_name}/{key}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def download_s3_object(bucket_name: str, key: str, download_path: str) -> str:
    """Download an object from S3.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key to download
        download_path: Local path where the file should be saved
    
    Returns:
        str: JSON string with download result
    """
    try:
        s3_client = get_s3_client()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        
        # Download file
        s3_client.download_file(bucket_name, key, download_path)
        
        # Get file size
        file_size = os.path.getsize(download_path)
        
        result = {
            'success': True,
            'bucket': bucket_name,
            'key': key,
            'download_path': download_path,
            'size_bytes': file_size,
            's3_url': f's3://{bucket_name}/{key}',
            'message': f'File downloaded successfully from s3://{bucket_name}/{key}'
        }
        
        logger.info(f"Downloaded s3://{bucket_name}/{key} to {download_path}")
        return json.dumps(result, indent=2)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            error_msg = f"Bucket {bucket_name} does not exist"
        elif error_code == 'NoSuchKey':
            error_msg = f"Object {key} does not exist in bucket {bucket_name}"
        else:
            error_msg = f"Failed to download s3://{bucket_name}/{key}: {str(e)}"
            
        logger.error(error_msg)
        return json.dumps({'error': error_msg})
        
    except Exception as e:
        error_msg = f"Failed to download s3://{bucket_name}/{key}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def delete_s3_object(bucket_name: str, key: str) -> str:
    """Delete an object from S3.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key to delete
    
    Returns:
        str: JSON string with deletion result
    """
    try:
        s3_client = get_s3_client()
        
        # Delete object
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        
        result = {
            'success': True,
            'bucket': bucket_name,
            'key': key,
            's3_url': f's3://{bucket_name}/{key}',
            'message': f'Object deleted successfully from s3://{bucket_name}/{key}'
        }
        
        logger.info(f"Deleted s3://{bucket_name}/{key}")
        return json.dumps(result, indent=2)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            error_msg = f"Bucket {bucket_name} does not exist"
        else:
            error_msg = f"Failed to delete s3://{bucket_name}/{key}: {str(e)}"
            
        logger.error(error_msg)
        return json.dumps({'error': error_msg})
        
    except Exception as e:
        error_msg = f"Failed to delete s3://{bucket_name}/{key}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def get_s3_object_info(bucket_name: str, key: str) -> str:
    """Get detailed information about an S3 object.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key
    
    Returns:
        str: JSON string with object metadata and information
    """
    try:
        s3_client = get_s3_client()
        
        # Get object metadata
        response = s3_client.head_object(Bucket=bucket_name, Key=key)
        
        result = {
            'bucket': bucket_name,
            'key': key,
            's3_url': f's3://{bucket_name}/{key}',
            'size_bytes': response.get('ContentLength', 0),
            'content_type': response.get('ContentType', 'unknown'),
            'last_modified': response.get('LastModified').isoformat() if response.get('LastModified') else None,
            'etag': response.get('ETag', '').strip('"'),
            'storage_class': response.get('StorageClass', 'STANDARD'),
            'metadata': response.get('Metadata', {}),
            'version_id': response.get('VersionId'),
            'server_side_encryption': response.get('ServerSideEncryption'),
            'cache_control': response.get('CacheControl'),
            'content_disposition': response.get('ContentDisposition'),
            'content_encoding': response.get('ContentEncoding'),
            'content_language': response.get('ContentLanguage'),
            'expires': response.get('Expires').isoformat() if response.get('Expires') else None
        }
        
        logger.info(f"Retrieved info for s3://{bucket_name}/{key}")
        return json.dumps(result, indent=2)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            error_msg = f"Bucket {bucket_name} does not exist"
        elif error_code == 'NoSuchKey':
            error_msg = f"Object {key} does not exist in bucket {bucket_name}"
        else:
            error_msg = f"Failed to get info for s3://{bucket_name}/{key}: {str(e)}"
            
        logger.error(error_msg)
        return json.dumps({'error': error_msg})
        
    except Exception as e:
        error_msg = f"Failed to get info for s3://{bucket_name}/{key}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})


@app.tool()
def read_s3_object_content(bucket_name: str, key: str, max_size_mb: int = 10) -> str:
    """Read the content of a text-based S3 object.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key to read
        max_size_mb: Maximum file size to read in MB (default 10MB for safety)
    
    Returns:
        str: JSON string with object content or error message
    """
    try:
        s3_client = get_s3_client()
        
        # First get object info to check size
        head_response = s3_client.head_object(Bucket=bucket_name, Key=key)
        object_size = head_response.get('ContentLength', 0)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if object_size > max_size_bytes:
            error_msg = f"Object size ({object_size} bytes) exceeds maximum allowed size ({max_size_bytes} bytes)"
            logger.warning(error_msg)
            return json.dumps({'error': error_msg})
        
        # Read object content
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response['Body'].read()
        
        # Try to decode as text
        try:
            content_str = content.decode('utf-8')
            content_type = 'text'
        except UnicodeDecodeError:
            try:
                content_str = content.decode('latin-1')
                content_type = 'text'
            except UnicodeDecodeError:
                # If it's binary content, return base64 encoded
                import base64
                content_str = base64.b64encode(content).decode('ascii')
                content_type = 'binary_base64'
        
        result = {
            'bucket': bucket_name,
            'key': key,
            's3_url': f's3://{bucket_name}/{key}',
            'content_type': content_type,
            'size_bytes': len(content),
            'content': content_str,
            'message': f'Successfully read content from s3://{bucket_name}/{key}'
        }
        
        logger.info(f"Read content from s3://{bucket_name}/{key} ({len(content)} bytes)")
        return json.dumps(result, indent=2)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            error_msg = f"Bucket {bucket_name} does not exist"
        elif error_code == 'NoSuchKey':
            error_msg = f"Object {key} does not exist in bucket {bucket_name}"
        else:
            error_msg = f"Failed to read content from s3://{bucket_name}/{key}: {str(e)}"
            
        logger.error(error_msg)
        return json.dumps({'error': error_msg})
        
    except Exception as e:
        error_msg = f"Failed to read content from s3://{bucket_name}/{key}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg}) 
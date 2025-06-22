"""AWS S3 Tools for Sherlog MCP Server

This module provides tools for interacting with Amazon S3 buckets and objects.
All operations are logged and can be accessed through audit endpoints.

Tools are only registered if AWS credentials are available.
"""

import json
import logging
import os

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

import pandas as pd

from sherlog_mcp.config import get_settings
from sherlog_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from sherlog_mcp.session import app

logger = logging.getLogger(__name__)


def _aws_credentials_available() -> bool:
    """Return True if AWS credentials are detectable via boto3's lookup chain."""
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except Exception:
        return False


if _aws_credentials_available():
    logger.info("AWS credentials detected - registering S3 tools")

    def get_s3_client():
        """Get configured S3 client with credentials from centralized config or AWS credential chain."""
        try:
            settings = get_settings()


            session_kwargs = {}

            if settings.aws_access_key_id and settings.aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = settings.aws_access_key_id
                session_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

                if settings.aws_session_token:
                    session_kwargs["aws_session_token"] = settings.aws_session_token

                logger.info("Using AWS credentials from configuration settings")
            else:
                logger.info("Using AWS default credential chain")

            session = boto3.Session(**session_kwargs)
            s3_client = session.client("s3", region_name=settings.aws_region)

            s3_client.list_buckets()
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

    def _list_s3_buckets_impl() -> pd.DataFrame:
        """Return S3 buckets as a DataFrame."""
        s3_client = get_s3_client()
        response = s3_client.list_buckets()

        rows = [
            {
                "name": bucket["Name"],
                "creation_date": bucket.get("CreationDate").isoformat()
                if bucket.get("CreationDate")
                else None,
            }
            for bucket in response.get("Buckets", [])
        ]

        df = pd.DataFrame(rows)
        return df

    def _list_s3_objects_impl(
        bucket_name: str, prefix: str = "", max_keys: int = 100
    ) -> pd.DataFrame:
        """Return objects in a bucket as DataFrame."""
        s3_client = get_s3_client()
        max_keys = min(max_keys, 1000)
        params = {"Bucket": bucket_name, "MaxKeys": max_keys}
        if prefix:
            params["Prefix"] = prefix

        response = s3_client.list_objects_v2(**params)

        rows = [
            {
                "bucket": bucket_name,
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj.get("LastModified").isoformat()
                if obj.get("LastModified")
                else None,
                "etag": obj.get("ETag", "").strip("\""),
                "storage_class": obj.get("StorageClass", "STANDARD"),
            }
            for obj in response.get("Contents", [])
        ]

        df = pd.DataFrame(rows)
        return df

    _SHELL.push(
        {
            "_list_s3_buckets_impl": _list_s3_buckets_impl,
            "_list_s3_objects_impl": _list_s3_objects_impl,
        }
    )

    @app.tool()
    async def list_s3_buckets(*, save_as: str = "s3_buckets") -> pd.DataFrame | None:
        """List S3 buckets and return a DataFrame.
        
        Args:
            save_as (str): Variable name to store the bucket list DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with columns 'name' and 'creation_date'
            
        Examples
        --------
        After calling this tool with save_as="s3_buckets":
        
        # View all buckets
        >>> execute_python_code("s3_buckets")
        
        # Get bucket names as list
        >>> execute_python_code("s3_buckets['name'].tolist()")
        
        # Sort by creation date (newest first)
        >>> execute_python_code("s3_buckets.sort_values('creation_date', ascending=False)")
        
        # Filter buckets by name pattern
        >>> execute_python_code("s3_buckets[s3_buckets['name'].str.contains('prod')]")
        
        # Count buckets
        >>> execute_python_code("len(s3_buckets)")
        """
        code = f"{save_as} = _list_s3_buckets_impl()\n{save_as}"
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def list_s3_objects(
        bucket_name: str,
        prefix: str = "",
        max_keys: int = 100,
        *,
        save_as: str = "s3_objects",
    ) -> pd.DataFrame | None:
        """List objects in an S3 bucket and return as DataFrame."""
        code = (
            f"{save_as} = _list_s3_objects_impl(\"{bucket_name}\", \"{prefix}\", {max_keys})\n"
            f"{save_as}"
        )
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _upload_s3_object_impl(
        bucket_name: str, key: str, file_path: str, content_type: str | None = None
    ) -> str:
        import mimetypes, json as _json
        s3_client = get_s3_client()
        if not os.path.exists(file_path):
            return _json.dumps({"error": f"Local file does not exist: {file_path}"})

        if not content_type:
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = "application/octet-stream"

        file_size = os.path.getsize(file_path)
        s3_client.upload_file(file_path, bucket_name, key, ExtraArgs={"ContentType": content_type})

        result = {
            "success": True,
            "bucket": bucket_name,
            "key": key,
            "file_path": file_path,
            "content_type": content_type,
            "size_bytes": file_size,
            "s3_url": f"s3://{bucket_name}/{key}",
        }
        return _json.dumps(result, indent=2)

    def _download_s3_object_impl(bucket_name: str, key: str, download_path: str) -> str:
        import json as _json
        s3_client = get_s3_client()
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        s3_client.download_file(bucket_name, key, download_path)
        file_size = os.path.getsize(download_path)
        return _json.dumps(
            {
                "success": True,
                "bucket": bucket_name,
                "key": key,
                "download_path": download_path,
                "size_bytes": file_size,
                "s3_url": f"s3://{bucket_name}/{key}",
            },
            indent=2,
        )

    def _delete_s3_object_impl(bucket_name: str, key: str) -> str:
        import json as _json
        s3_client = get_s3_client()
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        return _json.dumps(
            {
                "success": True,
                "bucket": bucket_name,
                "key": key,
                "s3_url": f"s3://{bucket_name}/{key}",
            },
            indent=2,
        )

    _SHELL.push(
        {
            "_upload_s3_object_impl": _upload_s3_object_impl,
            "_download_s3_object_impl": _download_s3_object_impl,
            "_delete_s3_object_impl": _delete_s3_object_impl,
        }
    )

    @app.tool()
    async def upload_s3_object(
        bucket_name: str,
        key: str,
        file_path: str,
        content_type: str | None = None,
        *,
        save_as: str = "s3_upload_result",
    ) -> str | None:
        code = (
            f'{save_as} = _upload_s3_object_impl("{bucket_name}", "{key}", "{file_path}", {repr(content_type)})\n'
            f"{save_as}"
        )
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def download_s3_object(
        bucket_name: str,
        key: str,
        download_path: str,
        *,
        save_as: str = "s3_download_result",
    ) -> str | None:
        code = (
            f'{save_as} = _download_s3_object_impl("{bucket_name}", "{key}", "{download_path}")\n'
            f"{save_as}"
        )
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def delete_s3_object(
        bucket_name: str,
        key: str,
        *,
        save_as: str = "s3_delete_result",
    ) -> str | None:
        code = (
            f'{save_as} = _delete_s3_object_impl("{bucket_name}", "{key}")\n'
            f"{save_as}"
        )
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

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

            response = s3_client.head_object(Bucket=bucket_name, Key=key)

            result = {
                "bucket": bucket_name,
                "key": key,
                "s3_url": f"s3://{bucket_name}/{key}",
                "size_bytes": response.get("ContentLength", 0),
                "content_type": response.get("ContentType", "unknown"),
                "last_modified": response.get("LastModified").isoformat()
                if response.get("LastModified")
                else None,
                "etag": response.get("ETag", "").strip('"'),
                "storage_class": response.get("StorageClass", "STANDARD"),
                "server_side_encryption": response.get("ServerSideEncryption"),
                "metadata": response.get("Metadata", {}),
            }

            return json.dumps(result, indent=2)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                error_msg = f"Bucket {bucket_name} does not exist"
            elif error_code == "NoSuchKey":
                error_msg = f"Object {key} does not exist in bucket {bucket_name}"
            else:
                error_msg = f"Failed to get info for s3://{bucket_name}/{key}: {str(e)}"

            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        except Exception as e:
            error_msg = f"Failed to get info for s3://{bucket_name}/{key}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

    @app.tool()
    def read_s3_object_content(
        bucket_name: str, key: str, max_size_mb: int = 10
    ) -> str:
        """Read the content of a text-based S3 object.

        Args:
            bucket_name: Name of the S3 bucket
            key: S3 object key to read
            max_size_mb: Maximum file size to read in MB (default 10MB)

        Returns:
            str: JSON string with object content and metadata

        """
        try:
            s3_client = get_s3_client()

            head_response = s3_client.head_object(Bucket=bucket_name, Key=key)
            size_bytes = head_response.get("ContentLength", 0)
            max_size_bytes = max_size_mb * 1024 * 1024

            if size_bytes > max_size_bytes:
                error_msg = f"Object size ({size_bytes} bytes) exceeds maximum allowed size ({max_size_bytes} bytes)"
                logger.error(error_msg)
                return json.dumps({"error": error_msg})

            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            content = response["Body"].read()

            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text_content = content.decode("latin-1")
                except UnicodeDecodeError:
                    text_content = f"[Binary content - {len(content)} bytes]"

            result = {
                "bucket": bucket_name,
                "key": key,
                "s3_url": f"s3://{bucket_name}/{key}",
                "size_bytes": size_bytes,
                "content_type": head_response.get("ContentType", "unknown"),
                "content": text_content,
                "last_modified": head_response.get("LastModified").isoformat()
                if head_response.get("LastModified")
                else None,
            }

            return json.dumps(result, indent=2)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                error_msg = f"Bucket {bucket_name} does not exist"
            elif error_code == "NoSuchKey":
                error_msg = f"Object {key} does not exist in bucket {bucket_name}"
            else:
                error_msg = (
                    f"Failed to read content from s3://{bucket_name}/{key}: {str(e)}"
                )

            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        except Exception as e:
            error_msg = (
                f"Failed to read content from s3://{bucket_name}/{key}: {str(e)}"
            )
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

else:
    logger.info("AWS credentials not detected - S3 tools will not be registered")

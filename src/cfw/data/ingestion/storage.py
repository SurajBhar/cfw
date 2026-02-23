"""
Storage abstraction for local filesystem and Azure Blob Storage.

Provides a unified interface for file operations across different storage backends,
enabling the same code to work with local files or cloud storage.
"""


import os
import shutil
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union, Iterator, BinaryIO
from fnmatch import fnmatch

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Provides a unified interface for file operations that can be implemented
    for different storage systems (local filesystem, Azure Blob, etc.).
    """

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        pass

    @abstractmethod
    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create a directory and all parent directories."""
        pass

    @abstractmethod
    def list_files(
        self,
        path: str,
        pattern: Optional[str] = None,
        recursive: bool = False
    ) -> List[str]:
        """
        List files in a directory.

        Args:
            path: Directory path to list
            pattern: Optional glob pattern to filter files (e.g., "*.png")
            recursive: If True, list files recursively

        Returns:
            List of file paths
        """
        pass

    @abstractmethod
    def list_dirs(self, path: str) -> List[str]:
        """List subdirectories in a directory."""
        pass

    @abstractmethod
    def copy_file(self, src: str, dst: str) -> None:
        """Copy a file from src to dst."""
        pass

    @abstractmethod
    def move_file(self, src: str, dst: str) -> None:
        """Move a file from src to dst."""
        pass

    @abstractmethod
    def delete_file(self, path: str) -> None:
        """Delete a file."""
        pass

    @abstractmethod
    def delete_dir(self, path: str, recursive: bool = False) -> None:
        """Delete a directory."""
        pass

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Read file contents as bytes."""
        pass

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a file."""
        pass

    @abstractmethod
    def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """Read file contents as text."""
        pass

    @abstractmethod
    def write_text(self, path: str, data: str, encoding: str = 'utf-8') -> None:
        """Write text to a file."""
        pass

    @abstractmethod
    def get_size(self, path: str) -> int:
        """Get file size in bytes."""
        pass

    @abstractmethod
    def join_path(self, *parts: str) -> str:
        """Join path components."""
        pass

    @abstractmethod
    def get_parent(self, path: str) -> str:
        """Get parent directory of a path."""
        pass

    @abstractmethod
    def get_filename(self, path: str) -> str:
        """Get filename from a path."""
        pass

    @abstractmethod
    def open_file(self, path: str, mode: str = 'rb') -> BinaryIO:
        """Open a file and return a file-like object."""
        pass


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend.

    Implements StorageBackend interface for local file operations
    using pathlib and standard library functions.
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize local storage.

        Args:
            base_path: Optional base path to prepend to all operations.
                      If None, paths are used as-is.
        """
        self.base_path = Path(base_path) if base_path else None

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base_path if set."""
        p = Path(path)
        if self.base_path and not p.is_absolute():
            return self.base_path / p
        return p

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        return self._resolve_path(path).exists()

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create a directory and all parent directories."""
        self._resolve_path(path).mkdir(parents=True, exist_ok=exist_ok)

    def list_files(
        self,
        path: str,
        pattern: Optional[str] = None,
        recursive: bool = False
    ) -> List[str]:
        """List files in a directory."""
        dir_path = self._resolve_path(path)
        if not dir_path.exists():
            return []

        if recursive:
            if pattern:
                files = list(dir_path.rglob(pattern))
            else:
                files = [f for f in dir_path.rglob('*') if f.is_file()]
        else:
            if pattern:
                files = list(dir_path.glob(pattern))
            else:
                files = [f for f in dir_path.iterdir() if f.is_file()]

        return [str(f) for f in files]

    def list_dirs(self, path: str) -> List[str]:
        """List subdirectories in a directory."""
        dir_path = self._resolve_path(path)
        if not dir_path.exists():
            return []
        return [str(d) for d in dir_path.iterdir() if d.is_dir()]

    def copy_file(self, src: str, dst: str) -> None:
        """Copy a file from src to dst."""
        src_path = self._resolve_path(src)
        dst_path = self._resolve_path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    def move_file(self, src: str, dst: str) -> None:
        """Move a file from src to dst."""
        src_path = self._resolve_path(src)
        dst_path = self._resolve_path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))

    def delete_file(self, path: str) -> None:
        """Delete a file."""
        file_path = self._resolve_path(path)
        if file_path.exists():
            file_path.unlink()

    def delete_dir(self, path: str, recursive: bool = False) -> None:
        """Delete a directory."""
        dir_path = self._resolve_path(path)
        if dir_path.exists():
            if recursive:
                shutil.rmtree(dir_path)
            else:
                dir_path.rmdir()

    def read_bytes(self, path: str) -> bytes:
        """Read file contents as bytes."""
        return self._resolve_path(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a file."""
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)

    def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """Read file contents as text."""
        return self._resolve_path(path).read_text(encoding=encoding)

    def write_text(self, path: str, data: str, encoding: str = 'utf-8') -> None:
        """Write text to a file."""
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(data, encoding=encoding)

    def get_size(self, path: str) -> int:
        """Get file size in bytes."""
        return self._resolve_path(path).stat().st_size

    def join_path(self, *parts: str) -> str:
        """Join path components."""
        return str(Path(*parts))

    def get_parent(self, path: str) -> str:
        """Get parent directory of a path."""
        return str(Path(path).parent)

    def get_filename(self, path: str) -> str:
        """Get filename from a path."""
        return Path(path).name

    def open_file(self, path: str, mode: str = 'rb') -> BinaryIO:
        """Open a file and return a file-like object."""
        return open(self._resolve_path(path), mode)


class AzureBlobStorage(StorageBackend):
    """
    Azure Blob Storage backend.

    Implements StorageBackend interface for Azure Blob Storage operations.
    Requires azure-storage-blob package.
    """

    def __init__(
        self,
        account_name: str,
        container_name: str,
        account_key: Optional[str] = None,
        sas_token: Optional[str] = None,
        connection_string: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Azure Blob Storage.

        Args:
            account_name: Azure storage account name
            container_name: Container name
            account_key: Account key for authentication (optional)
            sas_token: SAS token for authentication (optional)
            connection_string: Full connection string (optional)
            cache_dir: Local directory for caching downloads
        """
        try:
            from azure.storage.blob import BlobServiceClient, ContainerClient
        except ImportError:
            raise ImportError(
                "azure-storage-blob is required for Azure storage. "
                "Install with: pip install azure-storage-blob"
            )

        self.account_name = account_name
        self.container_name = container_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./data_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create blob service client
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                connection_string
            )
        elif sas_token:
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=sas_token
            )
        elif account_key:
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=account_key
            )
        else:
            raise ValueError(
                "Must provide one of: connection_string, sas_token, or account_key"
            )

        self.container_client = self.blob_service_client.get_container_client(
            container_name
        )

    def _normalize_path(self, path: str) -> str:
        """Normalize path for blob storage (use forward slashes)."""
        return path.replace('\\', '/').strip('/')

    def exists(self, path: str) -> bool:
        """Check if a blob exists."""
        blob_name = self._normalize_path(path)
        blob_client = self.container_client.get_blob_client(blob_name)
        try:
            blob_client.get_blob_properties()
            return True
        except Exception:
            # Check if it's a "directory" (prefix with blobs)
            blobs = list(self.container_client.list_blobs(
                name_starts_with=blob_name + '/',
                results_per_page=1
            ))
            return len(blobs) > 0

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """
        Create a directory marker in blob storage.

        Note: Azure Blob Storage doesn't have real directories, so this
        creates an empty marker blob or does nothing.
        """
        # Blob storage doesn't require explicit directory creation
        pass

    def list_files(
        self,
        path: str,
        pattern: Optional[str] = None,
        recursive: bool = False
    ) -> List[str]:
        """List blobs in a path."""
        prefix = self._normalize_path(path)
        if prefix and not prefix.endswith('/'):
            prefix += '/'

        files = []
        blobs = self.container_client.list_blobs(name_starts_with=prefix)

        for blob in blobs:
            blob_name = blob.name

            # Skip directory markers
            if blob_name.endswith('/'):
                continue

            # Handle non-recursive listing
            if not recursive:
                relative = blob_name[len(prefix):]
                if '/' in relative:
                    continue

            # Apply pattern filter
            if pattern:
                filename = blob_name.split('/')[-1]
                if not fnmatch(filename, pattern):
                    continue

            files.append(blob_name)

        return files

    def list_dirs(self, path: str) -> List[str]:
        """List subdirectories (prefixes) in a path."""
        prefix = self._normalize_path(path)
        if prefix and not prefix.endswith('/'):
            prefix += '/'

        dirs = set()
        blobs = self.container_client.list_blobs(name_starts_with=prefix)

        for blob in blobs:
            relative = blob.name[len(prefix):]
            if '/' in relative:
                subdir = relative.split('/')[0]
                dirs.add(prefix + subdir)

        return list(dirs)

    def copy_file(self, src: str, dst: str) -> None:
        """Copy a blob from src to dst."""
        src_blob = self._normalize_path(src)
        dst_blob = self._normalize_path(dst)

        src_client = self.container_client.get_blob_client(src_blob)
        dst_client = self.container_client.get_blob_client(dst_blob)

        dst_client.start_copy_from_url(src_client.url)

    def move_file(self, src: str, dst: str) -> None:
        """Move a blob from src to dst (copy then delete)."""
        self.copy_file(src, dst)
        self.delete_file(src)

    def delete_file(self, path: str) -> None:
        """Delete a blob."""
        blob_name = self._normalize_path(path)
        blob_client = self.container_client.get_blob_client(blob_name)
        try:
            blob_client.delete_blob()
        except Exception as e:
            logger.warning(f"Failed to delete blob {blob_name}: {e}")

    def delete_dir(self, path: str, recursive: bool = False) -> None:
        """Delete all blobs with the given prefix."""
        if not recursive:
            return

        prefix = self._normalize_path(path)
        if not prefix.endswith('/'):
            prefix += '/'

        blobs = self.container_client.list_blobs(name_starts_with=prefix)
        for blob in blobs:
            self.delete_file(blob.name)

    def read_bytes(self, path: str) -> bytes:
        """Read blob contents as bytes."""
        blob_name = self._normalize_path(path)
        blob_client = self.container_client.get_blob_client(blob_name)
        return blob_client.download_blob().readall()

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a blob."""
        blob_name = self._normalize_path(path)
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True)

    def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """Read blob contents as text."""
        return self.read_bytes(path).decode(encoding)

    def write_text(self, path: str, data: str, encoding: str = 'utf-8') -> None:
        """Write text to a blob."""
        self.write_bytes(path, data.encode(encoding))

    def get_size(self, path: str) -> int:
        """Get blob size in bytes."""
        blob_name = self._normalize_path(path)
        blob_client = self.container_client.get_blob_client(blob_name)
        properties = blob_client.get_blob_properties()
        return properties.size

    def join_path(self, *parts: str) -> str:
        """Join path components with forward slashes."""
        return '/'.join(p.strip('/') for p in parts if p)

    def get_parent(self, path: str) -> str:
        """Get parent directory of a path."""
        normalized = self._normalize_path(path)
        if '/' in normalized:
            return normalized.rsplit('/', 1)[0]
        return ''

    def get_filename(self, path: str) -> str:
        """Get filename from a path."""
        return self._normalize_path(path).split('/')[-1]

    def open_file(self, path: str, mode: str = 'rb') -> BinaryIO:
        """
        Open a blob and return a file-like object.

        For read operations, downloads to cache first.
        For write operations, returns a buffer that uploads on close.
        """
        import io
        import tempfile

        blob_name = self._normalize_path(path)

        if 'r' in mode:
            # Download to cache
            cache_path = self.cache_dir / blob_name.replace('/', '_')
            if not cache_path.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                blob_client = self.container_client.get_blob_client(blob_name)
                with open(cache_path, 'wb') as f:
                    f.write(blob_client.download_blob().readall())
            return open(cache_path, mode)
        else:
            # Return a buffer that uploads on close
            return _AzureWriteBuffer(self.container_client, blob_name)

    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a blob to local filesystem."""
        blob_name = self._normalize_path(remote_path)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob_client = self.container_client.get_blob_client(blob_name)
        with open(local_path, 'wb') as f:
            f.write(blob_client.download_blob().readall())

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a local file to blob storage."""
        blob_name = self._normalize_path(remote_path)
        blob_client = self.container_client.get_blob_client(blob_name)

        with open(local_path, 'rb') as f:
            blob_client.upload_blob(f, overwrite=True)


class _AzureWriteBuffer:
    """Buffer that uploads to Azure Blob on close."""

    def __init__(self, container_client, blob_name: str):
        import io
        self.container_client = container_client
        self.blob_name = blob_name
        self.buffer = io.BytesIO()

    def write(self, data: bytes) -> int:
        return self.buffer.write(data)

    def close(self) -> None:
        self.buffer.seek(0)
        blob_client = self.container_client.get_blob_client(self.blob_name)
        blob_client.upload_blob(self.buffer, overwrite=True)
        self.buffer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def create_storage_backend(cfg) -> StorageBackend:
    """
    Create a storage backend from configuration.

    Args:
        cfg: Configuration object with storage settings.
             Expected structure:
             - cfg.storage.backend: 'local' or 'azure'
             - For local: cfg.storage.paths.raw, cfg.storage.paths.processed, etc.
             - For azure: cfg.storage.connection.account_name, etc.

    Returns:
        Configured StorageBackend instance

    Example config (local):
        storage:
          backend: local
          paths:
            raw: ./data/raw
            processed: ./data/processed

    Example config (azure):
        storage:
          backend: azure
          connection:
            account_name: myaccount
            account_key: ${oc.env:AZURE_STORAGE_KEY}
            container_name: cfw-data
          paths:
            raw: data/raw
            processed: data/processed
    """
    backend_type = cfg.storage.backend.lower()

    if backend_type == 'local':
        base_path = cfg.storage.get('base_path', None)
        return LocalStorage(base_path=base_path)

    elif backend_type == 'azure':
        conn = cfg.storage.connection
        return AzureBlobStorage(
            account_name=conn.account_name,
            container_name=conn.container_name,
            account_key=conn.get('account_key', None),
            sas_token=conn.get('sas_token', None),
            connection_string=conn.get('connection_string', None),
            cache_dir=cfg.storage.get('cache_dir', './data_cache')
        )

    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")


def get_storage_paths(cfg) -> dict:
    """
    Get standardized storage paths from configuration.

    Args:
        cfg: Configuration with storage.paths section

    Returns:
        Dictionary with path keys: raw, processed, features, artifacts
    """
    paths = cfg.storage.paths
    return {
        'raw': paths.raw,
        'processed': paths.processed,
        'features': paths.get('features', './data/features'),
        'artifacts': paths.get('artifacts', './data/artifacts'),
    }

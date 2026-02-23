#!/usr/bin/env python3
"""Upload local folders to Azure Blob paths used by Azure ML datastores."""


from __future__ import annotations

import argparse
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local-to-blob synchronization."""
    parser = argparse.ArgumentParser(description="Sync local directory to Azure Blob")
    parser.add_argument("--source", required=True, help="Local source directory")
    parser.add_argument(
        "--target-path",
        required=True,
        help="Target blob prefix (e.g. cfw/processed/my_dataset/v20260211.1)",
    )
    parser.add_argument(
        "--container",
        default=os.getenv("AZURE_CONTAINER_NAME", "datadirectory"),
        help="Blob container name",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing blobs")
    return parser.parse_args()


def create_blob_service_client() -> BlobServiceClient:
    """Create a BlobServiceClient from connection string, keys, or managed identity."""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
    if not account_name:
        raise ValueError(
            "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT "
            "(plus auth via key/SAS/managed identity)."
        )

    account_key = os.getenv("AZURE_STORAGE_KEY")
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")

    if account_key:
        credential = account_key
    elif sas_token:
        credential = sas_token
    else:
        credential = DefaultAzureCredential()

    account_url = f"https://{account_name}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=credential)


def main() -> None:
    """Upload all files from a local directory to an Azure Blob prefix."""
    args = parse_args()

    source = Path(args.source).resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    blob_service_client = create_blob_service_client()
    container_client = blob_service_client.get_container_client(args.container)

    uploaded = 0
    prefix = args.target_path.strip("/")

    for local_file in source.rglob("*"):
        if not local_file.is_file():
            continue

        relative_path = local_file.relative_to(source).as_posix()
        blob_name = f"{prefix}/{relative_path}" if prefix else relative_path

        with open(local_file, "rb") as data:
            container_client.upload_blob(
                name=blob_name,
                data=data,
                overwrite=args.overwrite,
            )

        uploaded += 1
        print(f"Uploaded: {local_file} -> {args.container}/{blob_name}")

    print(f"Completed upload. Files uploaded: {uploaded}")


if __name__ == "__main__":
    main()

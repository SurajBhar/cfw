#!/usr/bin/env python3
"""Register versioned URI_FOLDER data assets in Azure ML.

Responsibilities:
- Provide CLI utilities for Azure ML job submission and asset registration.
- Standardize cloud dataset/environment wiring for repeatable experiments.
- Automate managed-compute workflows used in the paper release.
"""


from __future__ import annotations

import argparse
import os
from typing import Optional

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential


def parse_tags(values: list[str]) -> dict[str, str]:
    """Parse repeated `key=value` tags into a dictionary."""
    tags: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid tag format '{value}'. Use key=value")
        key, raw_val = value.split("=", 1)
        tags[key.strip()] = raw_val.strip()
    return tags


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Azure ML data-asset registration."""
    parser = argparse.ArgumentParser(description="Register Azure ML data assets")
    parser.add_argument("--name", required=True, help="Data asset name")
    parser.add_argument("--version", required=True, help="Data asset version")
    parser.add_argument("--path", required=True, help="URI folder path")
    parser.add_argument("--description", default="")
    parser.add_argument("--tag", action="append", default=[], help="Tag as key=value")
    parser.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace", default=os.getenv("AZUREML_WORKSPACE"))
    return parser.parse_args()


def require(value: Optional[str], name: str) -> str:
    """Return a required value or raise a descriptive error."""
    if value:
        return value
    raise ValueError(f"Missing required value: {name}")


def main() -> None:
    """Create or update a versioned Azure ML `URI_FOLDER` data asset."""
    args = parse_args()

    subscription_id = require(args.subscription_id, "subscription-id")
    resource_group = require(args.resource_group, "resource-group")
    workspace = require(args.workspace, "workspace")

    tags = parse_tags(args.tag)

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

    asset = Data(
        name=args.name,
        version=args.version,
        type=AssetTypes.URI_FOLDER,
        path=args.path,
        description=args.description,
        tags=tags,
    )

    created = ml_client.data.create_or_update(asset)
    print(f"Registered data asset: {created.name}:{created.version}")
    print(f"Path: {created.path}")


if __name__ == "__main__":
    main()

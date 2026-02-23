#!/usr/bin/env python3
"""Register Azure ML environment YAMLs for CFW workloads."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for environment registration."""
    parser = argparse.ArgumentParser(description="Register Azure ML environments")
    parser.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace", default=os.getenv("AZUREML_WORKSPACE"))
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "azureml/environments/env_dinov2_azureml.yaml",
            "azureml/environments/env_vitrans_azureml.yaml",
        ],
        help="Environment YAML files relative to cfw_new/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and print what would be registered without calling Azure.",
    )
    return parser.parse_args()


def require(value: Optional[str], name: str) -> str:
    """Return a required value or raise a descriptive error."""
    if value:
        return value
    raise ValueError(f"Missing required value: {name}")


def main() -> None:
    """Register one or more Azure ML environment YAML definitions."""
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]

    for rel_path in args.files:
        env_path = project_root / rel_path
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_path}")

    if args.dry_run:
        import yaml

        print("Dry run mode: no Azure API calls will be made.")
        for rel_path in args.files:
            env_path = project_root / rel_path
            data = yaml.safe_load(env_path.read_text(encoding="utf-8"))
            print(
                f"[DRY-RUN] Would register environment "
                f"{data.get('name')}:{data.get('version')} from {env_path}"
            )
        return

    from azure.ai.ml import MLClient, load_environment
    from azure.identity import DefaultAzureCredential

    subscription_id = require(args.subscription_id, "subscription-id")
    resource_group = require(args.resource_group, "resource-group")
    workspace = require(args.workspace, "workspace")

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

    for rel_path in args.files:
        env_path = project_root / rel_path
        environment = load_environment(source=str(env_path))
        created = ml_client.environments.create_or_update(environment)
        print(f"Registered environment: {created.name}:{created.version} from {env_path}")


if __name__ == "__main__":
    main()

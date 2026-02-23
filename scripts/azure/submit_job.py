#!/usr/bin/env python3
"""Submit an Azure ML job or pipeline YAML from cfw_new/azureml/."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Optional
import yaml


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Azure ML job submission."""
    parser = argparse.ArgumentParser(description="Submit Azure ML job from YAML")
    parser.add_argument("--file", required=True, help="Path to job YAML relative to cfw_new/")
    parser.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace", default=os.getenv("AZUREML_WORKSPACE"))
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream submitted job logs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate job YAML and print submission intent without calling Azure.",
    )
    parser.add_argument(
        "--set",
        dest="input_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override top-level YAML inputs at submit time (repeatable). "
            "Examples: --set trainer_profile=ddp_multi_node "
            "--set train_instance_count=2"
        ),
    )
    return parser.parse_args()


def require(value: Optional[str], name: str) -> str:
    """Return a required value or raise a descriptive error."""
    if value:
        return value
    raise ValueError(f"Missing required value: {name}")


def parse_input_overrides(raw_overrides: list[str]) -> dict[str, str]:
    """Parse repeated --set KEY=VALUE flags into a dictionary."""
    overrides: dict[str, str] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Expected KEY=VALUE format.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set value '{item}'. Key cannot be empty.")
        overrides[key] = value
    return overrides


def _coerce_scalar(value: str, declared_type: Optional[str]):
    """Coerce CLI string values based on AML input type when available."""
    t = (declared_type or "").lower()
    if t in {"integer", "int"}:
        return int(value)
    if t in {"number", "float", "double"}:
        return float(value)
    if t in {"boolean", "bool"}:
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        raise ValueError(
            f"Cannot parse boolean value '{value}' for input type '{declared_type}'."
        )
    return value


def _is_data_input(input_def: dict) -> bool:
    """Return True if an AML input definition points to a data/model path."""
    declared_type = str(input_def.get("type", "")).lower()
    if declared_type.startswith("uri_"):
        return True
    return declared_type in {"mltable", "custom_model", "mlflow_model", "triton_model"}


def apply_input_overrides(payload: dict, overrides: dict[str, str]) -> None:
    """Apply overrides to top-level YAML inputs in-place."""
    if not overrides:
        return
    if "inputs" not in payload or not isinstance(payload["inputs"], dict):
        raise ValueError("YAML does not define top-level 'inputs', cannot apply --set overrides.")

    inputs = payload["inputs"]
    unknown = [k for k in overrides if k not in inputs]
    if unknown:
        available = ", ".join(sorted(inputs.keys()))
        raise ValueError(
            "Unknown input override keys: "
            f"{', '.join(sorted(unknown))}. Available inputs: {available}"
        )

    for key, raw_value in overrides.items():
        current = inputs[key]
        if isinstance(current, dict):
            declared_type = current.get("type")
            if _is_data_input(current) or "path" in current:
                current["path"] = raw_value
            elif "default" in current:
                current["default"] = _coerce_scalar(raw_value, declared_type)
            else:
                inputs[key] = _coerce_scalar(raw_value, declared_type)
        else:
            inputs[key] = raw_value


def _summarize_input_value(value) -> str:
    if isinstance(value, dict):
        if "path" in value:
            return f"type={value.get('type', 'unknown')}, path={value['path']}"
        if "default" in value:
            return f"type={value.get('type', 'unknown')}, default={value['default']}"
    return str(value)


def main() -> None:
    """Load a job YAML, apply overrides, and submit it to Azure ML."""
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    job_path = project_root / args.file
    if not job_path.exists():
        raise FileNotFoundError(f"Job file not found: {job_path}")

    payload = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    input_overrides = parse_input_overrides(args.input_overrides)
    apply_input_overrides(payload, input_overrides)
    print(f"[INFO] Loaded job YAML: {job_path}", flush=True)

    if args.dry_run:
        print("Dry run mode: no Azure API calls will be made.")
        print(f"[DRY-RUN] Job file: {job_path}")
        print(f"[DRY-RUN] experiment_name: {payload.get('experiment_name')}")
        print(f"[DRY-RUN] compute: {payload.get('compute')}")
        print(f"[DRY-RUN] environment: {payload.get('environment')}")
        print(f"[DRY-RUN] command: {payload.get('command')}")
        if input_overrides:
            print(f"[DRY-RUN] applied overrides: {input_overrides}")
        if isinstance(payload.get("inputs"), dict):
            print("[DRY-RUN] inputs:")
            for key, value in payload["inputs"].items():
                print(f"  - {key}: {_summarize_input_value(value)}")
        return

    from azure.ai.ml import MLClient, load_job
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
    print(
        f"[INFO] Connected to workspace '{workspace}' (resource group '{resource_group}')",
        flush=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=".submit_job_",
        dir=str(job_path.parent),
        encoding="utf-8",
        delete=False,
    ) as tmp:
        yaml.safe_dump(payload, tmp, sort_keys=False)
        temp_job_path = Path(tmp.name)
    print(f"[INFO] Materialized temp job spec: {temp_job_path}", flush=True)

    try:
        print("[INFO] Loading AML job entity from YAML...", flush=True)
        job = load_job(source=str(temp_job_path))
        print(
            "[INFO] Submitting job to Azure ML (this may take time while packaging/uploading code)...",
            flush=True,
        )
        submit_start = time.time()
        created_job = ml_client.jobs.create_or_update(job)
        submit_elapsed = time.time() - submit_start
        print(f"Submitted job: {created_job.name}")
        print(f"[INFO] Submission completed in {submit_elapsed:.1f}s", flush=True)
        if input_overrides:
            print(f"Applied input overrides: {input_overrides}")

        studio_url = (
            f"https://ml.azure.com/runs/{created_job.name}"
            f"?wsid=/subscriptions/{subscription_id}/resourcegroups/{resource_group}"
            f"/workspaces/{workspace}"
        )
        print(f"Studio URL: {studio_url}")

        if args.stream:
            ml_client.jobs.stream(created_job.name)
    finally:
        temp_job_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

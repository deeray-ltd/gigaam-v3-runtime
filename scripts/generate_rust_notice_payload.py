#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Yuriy Krasilnikov
# Copyright (C) 2026 Deeray Ltd.
"""Generate the checked-in static Rust dependency notice payload.

The payload is deliberately derived from the locked, Linux service closure rather than from the
whole workspace. It copies only authoritative license/notice files packaged by each crate and
records checksum-bound crate archives for every package, including MPL-2.0 source availability.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tarfile
import tomllib
from typing import Any
from urllib.parse import quote


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT = WORKSPACE / "crates" / "service" / "licenses" / "rust"
TARGET = "x86_64-unknown-linux-gnu"
SERVICE = "gigaam-service"
NOTICE_PREFIXES = ("LICENSE", "COPYING", "NOTICE", "COPYRIGHT")
VARIANTS = {
    "CPU": (),
    "CUDA": ("--features", f"{SERVICE}/cuda"),
    "TensorRT": ("--features", f"{SERVICE}/tensorrt"),
}


class PayloadError(RuntimeError):
    """Raised when the lockfile, metadata, or registry source is not reproducible."""


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def cargo_metadata(feature_arguments: tuple[str, ...]) -> dict[str, Any]:
    command = [
        "cargo",
        "metadata",
        "--locked",
        "--offline",
        "--format-version",
        "1",
        "--filter-platform",
        TARGET,
        "--no-default-features",
        *feature_arguments,
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise PayloadError(
            "Cargo metadata failed for the locked service closure:\n"
            f"{result.stderr.rstrip()}"
        )
    return json.loads(result.stdout)


def proc_macro_only(package: dict[str, Any]) -> bool:
    library_kinds = {
        kind
        for target in package["targets"]
        for kind in target["kind"]
        if kind in {"lib", "proc-macro"}
    }
    return library_kinds == {"proc-macro"}


def normal_static_closure(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    packages = {package["id"]: package for package in metadata["packages"]}
    service_ids = [
        package_id
        for package_id, package in packages.items()
        if package["name"] == SERVICE and package["source"] is None
    ]
    if len(service_ids) != 1:
        raise PayloadError("Cargo metadata must contain exactly one local gigaam-service package")
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    pending = service_ids
    selected: set[str] = set()
    while pending:
        package_id = pending.pop()
        if package_id in selected:
            continue
        package = packages[package_id]
        if proc_macro_only(package):
            continue
        selected.add(package_id)
        pending.extend(
            dependency["pkg"]
            for dependency in nodes[package_id]["deps"]
            if any(kind["kind"] in (None, "normal") for kind in dependency["dep_kinds"])
        )
    return sorted(
        (packages[package_id] for package_id in selected if packages[package_id]["source"] is not None),
        key=lambda package: (package["name"], package["version"], package["source"]),
    )


def load_lock_checksums() -> dict[tuple[str, str, str], str]:
    lockfile = tomllib.loads((WORKSPACE / "Cargo.lock").read_text(encoding="utf-8"))
    checksums: dict[tuple[str, str, str], str] = {}
    for package in lockfile["package"]:
        source = package.get("source")
        checksum = package.get("checksum")
        if source is not None and checksum is not None:
            checksums[(package["name"], package["version"], source)] = checksum
    return checksums


def crate_archive(package: dict[str, Any], checksum: str) -> tarfile.TarFile:
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    archive_name = f"{package['name']}-{package['version']}.crate"
    candidates = sorted((cargo_home / "registry" / "cache").glob(f"*/{archive_name}"))
    if len(candidates) != 1:
        raise PayloadError(f"expected one cached crate archive for {package['id']}")
    content = candidates[0].read_bytes()
    if digest(content) != checksum:
        raise PayloadError(f"cached crate archive checksum does not match Cargo.lock for {package['id']}")
    return tarfile.open(fileobj=io.BytesIO(content), mode="r:gz")


def archive_member(archive: tarfile.TarFile, name: str, package: dict[str, Any]) -> tarfile.TarInfo:
    matches = [member for member in archive.getmembers() if member.name == name and member.isfile()]
    if len(matches) != 1:
        raise PayloadError(f"locked crate archive has no unique {name} for {package['id']}")
    return matches[0]


def archive_payload(package: dict[str, Any], checksum: str) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    with crate_archive(package, checksum) as archive:
        prefix = f"{package['name']}-{package['version']}/"
        manifest_member = archive_member(archive, f"{prefix}Cargo.toml", package)
        manifest_file = archive.extractfile(manifest_member)
        if manifest_file is None:
            raise PayloadError(f"locked crate archive cannot read Cargo.toml for {package['id']}")
        manifest = tomllib.loads(manifest_file.read().decode("utf-8"))
        package_metadata = manifest.get("package")
        if not isinstance(package_metadata, dict):
            raise PayloadError(f"locked crate archive has no package metadata for {package['id']}")
        if (
            package_metadata.get("name") != package["name"]
            or package_metadata.get("version") != package["version"]
        ):
            raise PayloadError(f"locked crate archive identity does not match Cargo.lock for {package['id']}")
        candidates = {
            member.name
            for member in archive.getmembers()
            if member.isfile()
            and member.name.startswith(prefix)
            and "/" not in member.name[len(prefix):]
            and Path(member.name).name.upper().startswith(NOTICE_PREFIXES)
        }
        license_file = package_metadata.get("license-file")
        if license_file is not None:
            relative = Path(license_file)
            if relative.is_absolute() or ".." in relative.parts:
                raise PayloadError(f"locked crate license file escapes its archive for {package['id']}")
            candidates.add(f"{prefix}{relative.as_posix()}")
        notices = []
        for candidate in sorted(candidates):
            member = archive_member(archive, candidate, package)
            content_file = archive.extractfile(member)
            if content_file is None:
                raise PayloadError(f"locked crate archive cannot read notice for {package['id']}: {candidate}")
            notices.append((candidate[len(prefix):], content_file.read()))
        return package_metadata, notices


def archive_url(package: dict[str, Any]) -> str:
    if package["source"] != "registry+https://github.com/rust-lang/crates.io-index":
        raise PayloadError(f"unsupported registry source for {package['id']}")
    name = quote(package["name"], safe="")
    version = quote(package["version"], safe="")
    return f"https://static.crates.io/crates/{name}/{name}-{version}.crate"


def locked_checksum(
    package: dict[str, Any], checksums: dict[tuple[str, str, str], str]
) -> str:
    key = (package["name"], package["version"], package["source"])
    checksum = checksums.get(key)
    if checksum is None:
        raise PayloadError(f"Cargo.lock lacks a checksum for {package['id']}")
    return checksum


def render_notice(
    packages: list[dict[str, Any]], checksums: dict[tuple[str, str, str], str]
) -> tuple[bytes, dict[str, bytes]]:
    lock_digest = digest((WORKSPACE / "Cargo.lock").read_bytes())
    closure_identity = "\n".join(
        "\t".join(
            (
                package["name"],
                package["version"],
                package["source"],
                locked_checksum(package, checksums),
            )
        )
        for package in packages
    )
    texts: dict[str, bytes] = {}
    records: list[tuple[dict[str, Any], dict[str, Any], str, list[tuple[str, str]]]] = []
    for package in packages:
        checksum = locked_checksum(package, checksums)
        notices = []
        package_metadata, archive_notices = archive_payload(package, checksum)
        for name, content in archive_notices:
            content_digest = digest(content)
            texts[content_digest] = content
            notices.append((name, content_digest))
        records.append((package, package_metadata, checksum, notices))
    lines = [
        "# GigaAM v3 Runtime Rust dependency notices",
        "",
        "This generated payload covers the exact registry crates statically linked into",
        "`gigaam-service` for the Linux release target. Do not edit it by hand; run",
        "`python3 scripts/generate_rust_notice_payload.py --check` to compare it with the",
        "locked Cargo metadata and checksum-verified local registry crate archives.",
        "",
        "## Scope",
        "",
        f"- Target: `{TARGET}`.",
        "- Service configurations: CPU (`--no-default-features`), CUDA",
        "  (`--no-default-features --features gigaam-service/cuda`), and TensorRT",
        "  (`--no-default-features --features gigaam-service/tensorrt`).",
        "- Selection: normal non-dev dependency edges only. Packages whose only library target",
        "  is a procedural macro are excluded because their code is executed during compilation,",
        "  not statically linked into `asr-serve`.",
        f"- Registry packages: {len(packages)}; the three selected closures are identical.",
        f"- Cargo.lock SHA-256: `{lock_digest}`.",
        f"- Closure SHA-256: `{digest(closure_identity.encode('utf-8'))}`.",
        "- Project workspace code is covered by the parent `COPYING` and `COPYING.LESSER` files.",
        "",
        "## Package records",
    ]
    for package, package_metadata, checksum, notices in records:
        lines.extend(("", f"### `{package['name']}` {package['version']}"))
        license_expression = package_metadata.get("license") or "not declared"
        lines.append(f"- License expression: `{license_expression}`.")
        lines.append(f"- Locked crate archive SHA-256: `{checksum}`.")
        lines.append(f"- Source archive: `{archive_url(package)}`.")
        repository = package_metadata.get("repository")
        if repository is not None:
            lines.append(f"- Repository declared by package metadata: `{repository}`.")
        authors = package_metadata.get("authors") or []
        if authors:
            lines.append(
                "- Authors declared by package metadata: "
                + "; ".join(f"`{author}`" for author in authors)
                + "."
            )
        if license_expression == "MPL-2.0":
            lines.append(
                "- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the "
                "checksum-bound source archive above."
            )
        if notices:
            lines.append("- Packaged license, permission, and copyright notices:")
            lines.extend(
                f"  - `{name}`: `texts/{content_digest}.txt` (SHA-256 `{content_digest}`)."
                for name, content_digest in notices
            )
        else:
            lines.append(
                "- The locked crate archive contains no standalone license or notice file; "
                "its metadata and source archive above are retained without inventing text."
            )
    return ("\n".join(lines) + "\n").encode("utf-8"), texts


def expected_payload() -> dict[Path, bytes]:
    checksums = load_lock_checksums()
    closures = {
        name: normal_static_closure(cargo_metadata(arguments))
        for name, arguments in VARIANTS.items()
    }
    identities = {
        name: [(package["name"], package["version"], package["source"]) for package in packages]
        for name, packages in closures.items()
    }
    if len({tuple(identity) for identity in identities.values()}) != 1:
        raise PayloadError(
            "CPU, CUDA, and TensorRT static service closures differ; "
            "split the payload before publishing"
        )
    notice, texts = render_notice(closures["CPU"], checksums)
    payload = {Path("NOTICE.md"): notice}
    for content_digest, content in texts.items():
        payload[Path("texts") / f"{content_digest}.txt"] = content
    return payload


def write_payload(payload: dict[Path, bytes]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="rust-license-payload-", dir=OUTPUT.parent
    ) as temporary:
        temporary_root = Path(temporary)
        replacement = temporary_root / "replacement"
        for relative, content in payload.items():
            destination = replacement / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        previous = temporary_root / "previous"
        moved_previous = False
        try:
            if OUTPUT.exists():
                os.replace(OUTPUT, previous)
                moved_previous = True
            os.replace(replacement, OUTPUT)
        except OSError as error:
            if moved_previous and not OUTPUT.exists():
                os.replace(previous, OUTPUT)
            raise PayloadError(f"cannot replace generated payload: {error}") from error


def check_payload(payload: dict[Path, bytes]) -> None:
    actual = (
        {
            path.relative_to(OUTPUT): path.read_bytes()
            for path in OUTPUT.rglob("*")
            if path.is_file()
        }
        if OUTPUT.is_dir()
        else {}
    )
    if actual == payload:
        return
    missing = sorted(str(path) for path in payload.keys() - actual.keys())
    extra = sorted(str(path) for path in actual.keys() - payload.keys())
    changed = sorted(
        str(path)
        for path in payload.keys() & actual.keys()
        if payload[path] != actual[path]
    )
    detail = [
        f"missing={','.join(missing) or '-'}",
        f"extra={','.join(extra) or '-'}",
        f"changed={','.join(changed) or '-'}",
    ]
    raise PayloadError("generated Rust notice payload is stale: " + "; ".join(detail))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the checked-in payload without writing")
    arguments = parser.parse_args()
    payload = expected_payload()
    if arguments.check:
        check_payload(payload)
    else:
        write_payload(payload)


if __name__ == "__main__":
    try:
        main()
    except PayloadError as error:
        raise SystemExit(f"error: {error}") from error

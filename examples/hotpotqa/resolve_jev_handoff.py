"""Resolve an exported Jev request on an internet-connected host."""

import argparse
import os
from pathlib import Path

from gepa.strategies.jev_constants import JEV_API_KEY_ENV
from gepa.strategies.jev_controller import JevController
from gepa.strategies.jev_handoff import load, resolve

try:
    # The Controller-only source does not include the diversity verifier.
    from gepa.strategies.jev_edit_verifier import JevEditVerifier  # pyright: ignore[reportMissingImports]
except ImportError:
    JevEditVerifier = None


def resolve_saved_request(request_path: Path, key: str) -> Path:
    """Resolve either supported role using its unchanged provider policy."""
    if not key:
        raise ValueError("Supply a protected local TypeSafe key file or TYPESAFE_API_KEY.")
    record = load(request_path)
    if record["namespace"] == JevController.JOURNAL_NAMESPACE:
        controller = JevController(api_key=key)
    elif JevEditVerifier is not None and record["namespace"] == JevEditVerifier.JOURNAL_NAMESPACE:
        controller = JevEditVerifier(api_key=key)
    else:
        raise ValueError("This checkout does not support the saved Jev role.")
    response = resolve(request_path, controller)
    if load(response)["error_type"] is not None:
        raise SystemExit("External request failed; its evidence is saved. Do not repeat it automatically.")
    return response


def main() -> None:
    """Write one sealed result without printing request content or credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    key = args.key_file.read_text().strip() if args.key_file else os.environ.get(JEV_API_KEY_ENV, "")
    response = resolve_saved_request(args.request, key)
    print(response)


if __name__ == "__main__":
    main()

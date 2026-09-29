"""Resolve an exported pilot request on the user's internet-connected computer."""

import argparse
import os
from pathlib import Path

from gepa.strategies.jev_controller import JevController
from gepa.strategies.jev_handoff import load, resolve

try:
    from gepa.strategies.jev_edit_verifier import JevEditVerifier
except ImportError:
    JevEditVerifier = None


def main() -> None:
    """Write one sealed result without printing request content or credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    key = args.key_file.read_text().strip() if args.key_file else os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise ValueError("Supply a protected local TypeSafe key file or TYPESAFE_API_KEY.")
    record = load(args.request)
    if record["namespace"] == JevController.JOURNAL_NAMESPACE:
        controller = JevController(api_key=key)
    elif JevEditVerifier is not None and record["namespace"] == JevEditVerifier.JOURNAL_NAMESPACE:
        controller = JevEditVerifier(api_key=key)
    else:
        raise ValueError("This checkout does not support the saved Jev role.")
    response = resolve(args.request, controller)
    if load(response)["error_type"] is not None:
        raise SystemExit("External request failed; its evidence is saved. Do not repeat it automatically.")
    print(response)


if __name__ == "__main__":
    main()

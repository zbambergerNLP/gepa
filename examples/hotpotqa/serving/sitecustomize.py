"""Install the opt-in repair in the vLLM server and spawned worker processes."""

import os
import sys
import traceback

if os.environ.get("GEPA_VLLM_SAFE_THINKING_BUDGET") == "1":
    try:
        from safe_thinking_budget import install

        print(f"GEPA reasoning-budget kernel repair: {install()}", file=sys.stderr, flush=True)
    except Exception:
        traceback.print_exc()
        # Python normally suppresses sitecustomize errors and continues startup.
        # An explicitly requested repair must never silently run unpatched.
        os._exit(78)

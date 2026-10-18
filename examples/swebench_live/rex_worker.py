"""Carry SWE-ReX runtime requests over inherited pipes inside the task container."""

import asyncio
import base64
import contextlib
import io
import json
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

from pydantic import TypeAdapter
from swerex.runtime import abstract
from swerex.runtime.local import LocalRuntime

REQUESTS = {
    "create_session": abstract.CreateSessionRequest,
    "run_in_session": abstract.Action,
    "close_session": abstract.CloseSessionRequest,
    "execute": abstract.Command,
    "read_file": abstract.ReadFileRequest,
    "write_file": abstract.WriteFileRequest,
}


async def main():
    """Delegate every operation to upstream LocalRuntime, including its persistent PTY."""
    with contextlib.redirect_stdout(sys.stderr):
        runtime = LocalRuntime()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method, payload = request["method"], request["payload"]
            with contextlib.redirect_stdout(sys.stderr):
                if method == "upload":
                    with tempfile.TemporaryDirectory() as temporary:
                        source = Path(temporary) / "upload"
                        source.mkdir()
                        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload["zip"]))) as archive:
                            archive.extractall(source)
                        result = await runtime.upload(
                            abstract.UploadRequest(source_path=str(source), target_path=payload["target_path"])
                        )
                elif method in REQUESTS:
                    result = await getattr(runtime, method)(TypeAdapter(REQUESTS[method]).validate_python(payload))
                elif method in {"is_alive", "close"}:
                    result = await getattr(runtime, method)()
                else:
                    raise ValueError("Unsupported runtime method")
            response = {"result": result.model_dump()}
        except Exception as error:
            response = {
                "error": abstract._ExceptionTransfer(
                    message=str(error),
                    class_path=type(error).__module__ + "." + type(error).__name__,
                    traceback=traceback.format_exc(),
                    extra_info=getattr(error, "extra_info", {}),
                ).model_dump()
            }
        print(json.dumps(response), flush=True)
        if request["method"] == "close":
            break


if __name__ == "__main__":
    asyncio.run(main())

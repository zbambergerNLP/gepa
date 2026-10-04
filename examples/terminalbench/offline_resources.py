"""Stage public, pinned task inputs without running solvers or reading grader data.

These commands run only while building an image with network access. Runtime
downloads use official caches or a hash-checked, exact-URL curl read-through.
Unknown URLs and unsupported curl options retain genuine curl behavior.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

ROOT = "/opt/harbor-offline"
SPEC_PATH = Path(__file__).with_name("terminalbench-v2.1-resources.json")
MTEB_REVISION = "71f6b6257025bbe06232352b86b09ab7bd7c904e"
MTEB_TASKS = (
    "BornholmBitextMining",
    "NorwegianCourtsBitextMining",
    "AngryTweetsClassification",
    "DanishPoliticalCommentsClassification",
    "DalajClassification",
    "DKHateClassification",
    "LccSentimentClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "NordicLangClassification",
    "NoRecClassification",
    "NorwegianParliamentClassification",
    "ScalaClassification",
    "SwedishSentimentClassification",
    "SweRecClassification",
    "DanFeverRetrieval",
    "NorQuadRetrieval",
    "SNLRetrieval",
    "SwednRetrieval",
    "SweFaqRetrieval",
    "TV2Nordretrieval",
    "TwitterHjerneRetrieval",
    "SNLHierarchicalClusteringS2S",
    "SNLHierarchicalClusteringP2P",
    "SwednClusteringP2P",
    "SwednClusteringS2S",
    "VGHierarchicalClusteringS2S",
    "VGHierarchicalClusteringP2P",
)
UV_ASSETS = {
    "0.9.5": {
        "uv-installer.sh": "8402ab80d2ef54d7044a71ea4e4e1e8db3b20c87c7bffbc30bff59f1e80ebbd5",
        "uv-x86_64-unknown-linux-gnu.tar.gz": "2cf10babba653310606f8b49876cfb679928669e7ddaa1fb41fb00ce73e64f66",
        "uv-x86_64-unknown-linux-gnu.tar.gz.sha256": "08e0a94777ab9d664f0ab4c10e4b66ebe124ec71d82978c9021a0b8cd6ca3b37",
    },
    "0.8.15": {
        "uv-installer.sh": "b64cc9fb212949d3fb88ac05833949dbb739a1c74f71070fe353d1adbd46b082",
        "uv-x86_64-unknown-linux-gnu.tar.gz": "be9878e9d08ebcb621a683aba52e7fb8bbf92b2532e0d759026ffcc067673042",
        "uv-x86_64-unknown-linux-gnu.tar.gz.sha256": "2899367da565333b4cbd4a98e602e5c8af215a8d3728a0fd72a3a448dafa2700",
    },
}

# Keep this executable compatible with the base image's Python 3.9. The managed
# Harbor environment does not exist yet when its genuine uv installer runs.
CURL_WRAPPER = r"""#!/usr/bin/python3
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

REAL_CURL = "/opt/harbor-offline/original-curl"
URL_MAP = "/opt/harbor-offline/resource-urls.json"


def main():
    original = sys.argv[1:]
    args = list(original)
    positions = []
    protocols = []
    flags = {
        "--fail", "--fail-with-body", "--location", "--silent", "--show-error",
        "--remote-name", "--create-dirs", "--no-progress-meter", "--progress-bar",
        "--tlsv1.2", "--tlsv1.3", "--compressed", "--insecure", "--retry-all-errors",
    }
    values = {
        "--output", "--output-dir", "--connect-timeout", "--max-time", "--retry",
        "--retry-delay", "--retry-max-time", "--user-agent", "--proxy", "--noproxy",
    }
    i = 0
    try:
        while i < len(args):
            arg = args[i]
            if arg == "--":
                positions.extend((j, "") for j in range(i + 1, len(args)))
                break
            if arg in ("--url", "--proto", "--proto-redir"):
                i += 1
                if arg == "--url":
                    positions.append((i, ""))
                else:
                    protocols.append((i, "", arg))
            elif arg.startswith("--url="):
                positions.append((i, "--url="))
            elif arg.startswith(("--proto=", "--proto-redir=")):
                option = arg.split("=", 1)[0]
                protocols.append((i, option + "=", option))
            elif arg in flags:
                pass
            elif arg in values:
                i += 1
                if i >= len(args):
                    raise ValueError("missing option argument")
            elif any(arg.startswith(option + "=") for option in values):
                pass
            elif arg.startswith("--"):
                raise ValueError("unsupported option")
            elif arg.startswith("-") and arg != "-":
                cluster = arg[1:]
                while cluster:
                    flag, cluster = cluster[0], cluster[1:]
                    if flag in "fLsSkO#":
                        continue
                    if flag in "oAmx":
                        if not cluster:
                            i += 1
                            if i >= len(args):
                                raise ValueError("missing option argument")
                        cluster = ""
                        continue
                    raise ValueError("unsupported option")
            else:
                positions.append((i, ""))
            i += 1
        mapping = json.loads(Path(URL_MAP).read_text())
        urls = [args[j][len(prefix):] for j, prefix in positions]
        if not urls or any(url not in mapping for url in urls):
            raise ValueError("uncached URL")
        for index, prefix, option in protocols:
            if option != "--proto":
                continue
            for url in urls:
                scheme = urlsplit(url).scheme
                allowed = True
                for protocol in args[index][len(prefix):].split(","):
                    modifier = protocol[:1] if protocol[:1] in "+-=" else "+"
                    name = protocol[1:] if protocol[:1] in "+-=" else protocol
                    if modifier == "=":
                        allowed = False
                    if name in (scheme, "all"):
                        allowed = modifier != "-"
                if not allowed:
                    raise ValueError("original URL prohibited by protocol policy")
    except (ValueError, OSError, IndexError):
        os.execv(REAL_CURL, [REAL_CURL] + original)

    # Preflight every input before genuine curl can truncate an output file.
    for (index, prefix), url in zip(positions, urls):
        record = mapping[url]
        path = Path(record["path"])
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError("SHA256 mismatch")
        except (OSError, ValueError) as error:
            print("curl: (26) pinned resource cache invalid: %s: %s" % (url, error), file=sys.stderr)
            return 26
        args[index] = prefix + path.absolute().as_uri()
    # An installer may restrict curl to HTTPS. Only all-cached calls reach this
    # point; adding file cannot relax the protocol policy for an unknown URL.
    for index, prefix, option in protocols:
        args[index] = args[index] + ",file"
    os.execv(REAL_CURL, [REAL_CURL] + args)


if __name__ == "__main__":
    sys.exit(main())
"""

WINDOWS_LOG_RESET = r"""(
set -eu
test -d /var/log/nginx
test ! -L /var/log/nginx
test "$(find /var/log/nginx -mindepth 1 -maxdepth 1 -printf '.')" = '..'
for harbor_nginx_log in /var/log/nginx/access.log /var/log/nginx/error.log; do
    test -f "$harbor_nginx_log"
    test ! -L "$harbor_nginx_log"
    test ! -s "$harbor_nginx_log"
done
rm /var/log/nginx/access.log /var/log/nginx/error.log
rmdir /var/log/nginx
mkdir -m 0755 /var/log/nginx
(umask 027; : > /var/log/nginx/access.log; : > /var/log/nginx/error.log)
)
"""

WINDOWS_STARTUP = r"""# Sourced only by noninteractive bash inside the prepared task image.
if [ "${BASH_EXECUTION_STRING-}" = 'exec /staging/bootstrap.sh "$@"' ] &&
   [ "${2-}" = /staging/_hbexec.py ] &&
   [ -f /staging/bootstrap.sh ] && [ -f /staging/_hbexec.py ]; then
    unset BASH_ENV
    (
        exec 9>/run/harbor-windows-startup.lock
        flock -x 9
        if [ -s /run/harbor-windows-startup.pid ]; then
            harbor_supervisor_pid=$(cat /run/harbor-windows-startup.pid)
            if kill -0 "$harbor_supervisor_pid" 2>/dev/null; then exit 0; fi
        fi
        supervisord -c /etc/supervisor/supervisord.conf > /root/supervisord-bootstrap.log 2>&1 9>&- &
        harbor_supervisor_pid=$!
        printf '%s\n' "$harbor_supervisor_pid" > /run/harbor-windows-startup.pid
        harbor_start_attempt=0
        while [ "$harbor_start_attempt" -lt 100 ]; do
            if ! kill -0 "$harbor_supervisor_pid" 2>/dev/null; then
                echo 'Official Windows task supervisord command failed; inspect /root/supervisord-bootstrap.log' >&2
                cat /root/supervisord-bootstrap.log >&2
                exit 1
            fi
            if [ -s /root/supervisord.pid ] &&
               [ "$(cat /root/supervisord.pid)" = "$harbor_supervisor_pid" ]; then exit 0; fi
            harbor_start_attempt=$((harbor_start_attempt + 1))
            sleep 0.1
        done
        echo 'Official Windows task supervisord command did not become ready' >&2
        exit 1
    ) || exit 1
fi
"""


def _write(path: str, content: str, *, append: bool = False) -> str:
    if "\nFOREST_RESOURCE_EOF\n" in content:
        raise ValueError("Reserved resource heredoc delimiter")
    redirect = ">>" if append else ">"
    return f"cat {redirect} {shlex.quote(path)} <<'FOREST_RESOURCE_EOF'\n{content.rstrip()}\nFOREST_RESOURCE_EOF"


def _python(program: str, executable: str = "/usr/bin/python3") -> str:
    return f"{shlex.quote(executable)} - <<'FOREST_PYTHON_EOF'\n{program.rstrip()}\nFOREST_PYTHON_EOF"


def _common_commands() -> list[str]:
    return [
        f"mkdir -p {ROOT}/resource-bin {ROOT}/resources {ROOT}/wheels",
        f"test -f {ROOT}/resource-urls.json || printf '{{}}\\n' > {ROOT}/resource-urls.json",
        f"test -f {ROOT}/original-curl || cp -p /usr/bin/curl {ROOT}/original-curl",
        _write(f"{ROOT}/resource-bin/curl", CURL_WRAPPER),
        f"chmod 755 {ROOT}/resource-bin/curl",
        # Harbor prepends /usr/bin to every agent/verifier command's PATH.
        # Both entrypoints must use the same cache while the original remains
        # untouched across the installer's and task's repeated setup calls.
        f"cp {ROOT}/resource-bin/curl /usr/bin/curl",
        _write(
            f"{ROOT}/resource-env.sh",
            f"export PATH={ROOT}/resource-bin:$PATH\n"
            f'export PIP_FIND_LINKS="{ROOT}/wheels${{PIP_FIND_LINKS:+ $PIP_FIND_LINKS}}"\n'
            f'export UV_FIND_LINKS="{ROOT}/wheels${{UV_FIND_LINKS:+ $UV_FIND_LINKS}}"',
        ),
        _write(
            f"{ROOT}/README.resources.txt",
            "Public offline prerequisites are stored here. No task solutions are prepared.\n"
            "resource-policy.json describes this task's audited inputs. resource-urls.json records the exact public "
            "URLs, source revisions, SHA256 hashes, and local paths.\n"
            "resources/ contains untouched public source archives and pinned Git mirrors. Exact declared Git URLs "
            "use repository-scoped system Git rewrites; declared curl downloads use verified local bytes.\n"
            "wheels/ and solver-requirements.txt contain Python dependencies available through PIP_FIND_LINKS and "
            "UV_FIND_LINKS. apt/ contains Debian packages used by normal apt-get install.\n"
            "huggingface/ is the standard HF_HOME cache. mteb/ is the standard MTEB_CACHE with complete relevant "
            "historical coverage, when applicable. cran/ is a local source repository for R package installation.\n"
            "The task's required source builds, installations, configurations, training, and output files remain "
            "the solver's responsibility.",
        ),
        f"export PATH={ROOT}/resource-bin:$PATH",
    ]


def _fetch(
    url: str,
    relative_path: str,
    *,
    revision: str,
    sha256: str | None = None,
    md5: str | None = None,
    aliases: tuple[str, ...] = (),
) -> list[str]:
    """Keep original public bytes and a provenance record, using genuine curl."""
    path = f"{ROOT}/resources/{relative_path}"
    data = {"url": url, "path": path, "revision": revision, "sha256": sha256, "md5": md5, "aliases": list(aliases)}
    return [
        _python(f"""import hashlib, json, os, subprocess
from pathlib import Path
record = {data!r}
path = Path(record["path"])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(path.name + ".partial")
subprocess.run(["{ROOT}/original-curl", "--fail", "--location", "--silent", "--show-error",
                "--retry", "3", record["url"], "--output", str(temporary)], check=True)
digests = {{name: hashlib.new(name) for name in ("sha256", "md5")}}
with temporary.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        for digest in digests.values(): digest.update(block)
for name, digest in digests.items():
    if record.get(name) and digest.hexdigest() != record[name]:
        raise RuntimeError("Pinned public resource hash mismatch: " + record["url"])
os.replace(temporary, path)
mapping_path = Path("{ROOT}/resource-urls.json")
mapping = json.loads(mapping_path.read_text())
for url in [record["url"]] + record["aliases"]:
    # curl -O takes its filename from the rewritten URL. Retain each URL's
    # original basename even when aliases share the same byte-identical input.
    from urllib.parse import urlsplit
    basename = Path(urlsplit(url).path).name
    alias_path = path
    if basename != path.name:
        alias_path = path.parent / basename
        if alias_path.exists() or alias_path.is_symlink(): alias_path.unlink()
        alias_path.symlink_to(path.name)
    mapping[url] = {{"path": str(alias_path), "sha256": digests["sha256"].hexdigest(),
                    "source_url": record["url"], "revision": record["revision"]}}
mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\\n")
""")
    ]


def uv_installer_recipe_commands(version: str) -> list[str]:
    """Cache the authentic Linux x86_64 uv installer and every nested release file."""
    if version not in UV_ASSETS:
        raise ValueError(f"No audited uv installer release: {version}")
    commands = _common_commands()
    commands.append('test "$(uname -m)" = x86_64 || { echo "uv resource cache requires Linux x86_64" >&2; exit 1; }')
    for filename, digest in UV_ASSETS[version].items():
        commands.extend(
            _fetch(
                f"https://github.com/astral-sh/uv/releases/download/{version}/{filename}",
                f"uv/{version}/{filename}",
                revision=version,
                sha256=digest,
                aliases=(f"https://astral.sh/uv/{version}/install.sh",) if filename == "uv-installer.sh" else (),
            )
        )
    return commands


def _git_mirror(
    name: str, url: str, revision: str, *, ref: str | None = None, aliases: tuple[str, ...] = ()
) -> list[str]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Public source mirrors require a full commit SHA")
    path = f"{ROOT}/resources/{name}.git"
    target = ref or "refs/heads/frozen"
    commands = [
        "apt-get install -y --no-install-recommends git",
        f"git init --bare {shlex.quote(path)}",
        shlex.join(["git", f"--git-dir={path}", "fetch", "--depth", "1", url, f"{ref or revision}:{target}"]),
        f'test "$(git --git-dir={path} rev-parse {shlex.quote(target + "^{commit}")})" = {revision}',
        f"git --git-dir={path} update-ref refs/heads/frozen {revision}",
        f"git --git-dir={path} symbolic-ref HEAD refs/heads/frozen",
        "git config --system protocol.file.allow always",
        _write(f"{ROOT}/resources/{name}-source.json", json.dumps({"url": url, "revision": revision, "ref": ref})),
    ]
    for alias in dict.fromkeys((url, *aliases)):
        commands.append(shlex.join(["git", "config", "--system", "--add", f"url.file://{path}.insteadOf", alias]))
    return commands


def _apt_cache(packages: list[str]) -> list[str]:
    return [
        f"mkdir -p {ROOT}/apt/partial",
        _write("/etc/apt/apt.conf.d/99harbor-resource-cache", f'Dir::Cache::archives "{ROOT}/apt";'),
        shlex.join(["apt-get", "--download-only", "install", "-y", "--no-install-recommends", *packages]),
        f"find {ROOT}/apt -name '*.deb' -type f -exec sha256sum '{{}}' + > {ROOT}/apt-packages.sha256",
    ]


def _preserve_nproc_wrapper(expected: int) -> list[str]:
    # Harbor puts /usr/bin ahead of the task's deliberate /usr/local/bin limit.
    return [
        f'test "$(/usr/local/bin/nproc)" = {expected}',
        f"cp -p /usr/bin/nproc {ROOT}/original-nproc",
        "cp -p /usr/local/bin/nproc /usr/bin/nproc",
    ]


def _python_cache(packages: list[str]) -> list[str]:
    target = f"{ROOT}/solver-dependencies"
    return [
        f'uv venv --seed --python "$(command -v python3)" {target}',
        shlex.join(["uv", "pip", "install", "--python", f"{target}/bin/python", *packages]),
        shlex.join([f"{target}/bin/python", "-m", "pip", "wheel", "--wheel-dir", f"{ROOT}/wheels", *packages]),
        f"uv pip freeze --python {target}/bin/python > {ROOT}/solver-requirements.txt",
        f"find {ROOT}/wheels -type f -exec sha256sum '{{}}' + > {ROOT}/solver-wheels.sha256",
    ]


def _hf_snapshots(snapshots: list[dict], *, dataset: bool = False) -> list[str]:
    commands = [
        _write(
            f"{ROOT}/resource-env.sh",
            f"export HF_HOME={ROOT}/huggingface\nexport HF_HUB_OFFLINE=1\nexport HF_DATASETS_OFFLINE=1",
            append=True,
        ),
        f"export HF_HOME={ROOT}/huggingface",
        f"uv venv --python python3 {ROOT}/resource-loader",
        f"uv pip install --python {ROOT}/resource-loader/bin/python 'huggingface-hub==0.34.4' 'datasets==3.6.0'",
        _python(
            f"""import hashlib, json, os
from pathlib import Path
from huggingface_hub import snapshot_download
records = {snapshots!r}
for record in records:
    path = Path(snapshot_download(repo_id=record["repo_id"], revision=record["revision"],
                repo_type=record.get("repo_type", "model"), ignore_patterns=record.get("ignore_patterns")))
    repo = path.parent.parent
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs" / "main").write_text(record["revision"])
    for alias in record.get("aliases", []):
        alias_path = repo.parent / ("models--" + alias.replace("/", "--"))
        if not alias_path.exists(): alias_path.symlink_to(repo.name)
    files = {{}}
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
            files[file.relative_to(path).as_posix()] = digest.hexdigest()
    if not files: raise RuntimeError("Empty pinned Hugging Face snapshot")
    record["files"] = files
Path("{ROOT}/hf-snapshots.json").write_text(json.dumps(records, indent=2) + "\\n")
""",
            f"{ROOT}/resource-loader/bin/python",
        ),
    ]
    if dataset:
        commands.append(
            _python(
                """import datasets
repo_id = "ryanmarten/OpenThoughts-1k-sample"
revision = "a82400884621626d41bef89b7604f8054e7e00e0"
configs = datasets.get_dataset_config_names(repo_id, revision=revision)
if not configs: raise RuntimeError("Pinned dataset has no configurations")
for config in configs:
    datasets.load_dataset(repo_id, name=config, revision=revision)
""",
                f"{ROOT}/resource-loader/bin/python",
            )
        )
        commands.append(
            f"HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 {ROOT}/resource-loader/bin/python -c "
            + shlex.quote("import datasets; d=datasets.load_dataset('ryanmarten/OpenThoughts-1k-sample'); assert d")
        )
    return commands


def _mteb_results() -> list[str]:
    path = f"{ROOT}/mteb/results"
    filenames = ["model_meta.json", *(task + ".json" for task in MTEB_TASKS)]
    return [
        "apt-get install -y --no-install-recommends git",
        f"mkdir -p {ROOT}/mteb",
        f"git init {path}",
        f"git -C {path} remote add origin https://github.com/embeddings-benchmark/results",
        f"git -C {path} fetch --filter=blob:none --depth 1 origin {MTEB_REVISION}",
        f"git -C {path} sparse-checkout init --no-cone",
        _write(f"{path}/.git/info/sparse-checkout", "\n".join("**/" + name for name in filenames)),
        f"git -C {path} checkout -b frozen FETCH_HEAD",
        _python(f"""import hashlib, json, subprocess
from pathlib import Path
root = Path({path!r})
names = set({filenames!r})
paths = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "--name-only", "-z", "HEAD"]).decode().split("\\0")
selected = sorted(p for p in paths if p and Path(p).name in names)
if not selected: raise RuntimeError("Historical MTEB tree is empty")
missing = names - {{Path(p).name for p in selected}}
if missing: raise RuntimeError("Historical MTEB tasks missing: " + repr(sorted(missing)))
files = {{}}
for name in selected:
    content = (root / name).read_bytes()
    files[name] = hashlib.sha256(content).hexdigest()
Path("{ROOT}/mteb-snapshot.json").write_text(json.dumps({{"source_url": "https://github.com/embeddings-benchmark/results", "revision": {MTEB_REVISION!r}, "scope": "all model metadata and all 28 Scandinavian task results across every model and revision", "files": files}}, indent=2) + "\\n")
"""),
        # The official cache calls git pull. Its upstream is the same frozen
        # local branch, so that operation succeeds without contacting GitHub.
        f"git -C {path} remote set-url origin file://{path}",
        f"git -C {path} config branch.frozen.remote origin",
        f"git -C {path} config branch.frozen.merge refs/heads/frozen",
        f"git -C {path} config remote.origin.fetch '+refs/heads/frozen:refs/remotes/origin/frozen'",
        f"GIT_TERMINAL_PROMPT=0 git -C {path} pull --ff-only",
        _write(f"{ROOT}/resource-env.sh", f"export MTEB_CACHE={ROOT}/mteb", append=True),
    ]


def _rstan_cache() -> list[str]:
    return _apt_cache(["r-base-dev", "build-essential", "libtbb-dev", "pandoc"]) + [
        f"mkdir -p {ROOT}/cran/src/contrib",
        "Rscript --vanilla - <<'FOREST_R_EOF'\n"
        'repo <- "https://packagemanager.posit.co/cran/2025-08-31"\n'
        'db <- available.packages(repos=repo, type="source")\n'
        'stopifnot(db["rstan", "Version"] == "2.32.7")\n'
        'deps <- tools::package_dependencies("rstan", db=db, which=c("Depends", "Imports", "LinkingTo"), recursive=TRUE)[[1]]\n'
        'packages <- unique(c("rstan", deps[deps %in% rownames(db)]))\n'
        f'downloaded <- download.packages(packages, destdir="{ROOT}/cran/src/contrib", repos=repo, available=db, type="source")\n'
        "stopifnot(setequal(downloaded[,1], packages))\n"
        f'tools::write_PACKAGES("{ROOT}/cran/src/contrib", type="source")\n'
        f'write.table(db[packages, c("Package", "Version")], "{ROOT}/cran-versions.tsv", sep="\\t", quote=FALSE, row.names=FALSE)\n'
        "FOREST_R_EOF",
        f"find {ROOT}/cran -type f -exec sha256sum '{{}}' + > {ROOT}/cran.sha256",
        _write(f"{ROOT}/Rprofile", f'options(repos=c(CRAN="file://{ROOT}/cran"))'),
        _write(f"{ROOT}/resource-env.sh", f"export R_PROFILE_USER={ROOT}/Rprofile", append=True),
    ]


def resource_recipe_commands(task_id: str, task_ref: str) -> list[str]:
    """Return audited infrastructure commands for an exact immutable task package."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    entry = spec["tasks"].get(task_id)
    if not entry or entry["task_ref"] != task_ref:
        raise ValueError(f"No audited public-resource recipe for {task_id}@{task_ref}")
    kind = entry["recipe"]
    commands = _common_commands()
    commands.append(_write(f"{ROOT}/resource-policy.json", json.dumps(entry, indent=2)))
    if kind == "base_and_verifier_only":
        return commands
    if kind == "ocaml":
        commands += _git_mirror(
            "ocaml",
            "https://github.com/sadiqj/ocaml/",
            "356d558bf89552ed9229253ec4135c691234f060",
            ref="refs/tags/tag_purposefully_broken_sweeping_changes",
        )
    elif kind == "pyknotid":
        commands += _git_mirror(
            "pyknotid",
            "https://github.com/SPOCKnots/pyknotid.git",
            "441c807dbec2ee32e1da572e24e58d52a4eb7afa",
            ref="refs/tags/0.5.3",
            aliases=("https://github.com/SPOCKnots/pyknotid",),
        )
        commands += _python_cache(
            [
                "numpy==2.3.0",
                "Cython==3.1.2",
                "setuptools==80.9.0",
                "wheel==0.45.1",
                "networkx",
                "planarity",
                "peewee",
                "vispy",
                "sympy",
                "appdirs",
                "requests",
                "tqdm",
            ]
        )
    elif kind == "pmars":
        commands += _apt_cache(["build-essential", "libncurses-dev", "dpkg-dev"])
        for filename, digest in {
            "pmars_0.9.4-1.dsc": None,
            "pmars_0.9.4.orig.tar.xz": "d0467d602c37af6e887ac81f4cbb812d090224411196d3401023843630fdd00d",
            "pmars_0.9.4-1.debian.tar.xz": "ca468a97c0fd603e59e0e7c48ef7c9f24a8ac66b5d8c6d5061c4567ccedabfe1",
        }.items():
            commands += _fetch(
                "https://deb.debian.org/debian/pool/main/p/pmars/" + filename,
                "pmars/" + filename,
                revision="0.9.4-1",
                sha256=digest,
            )
    elif kind == "povray":
        commands += _apt_cache(["build-essential", "unzip"])
        for filename in ("POVSRC.ZIP", "POVDOC.ZIP", "POVSCN.ZIP", "README2.2", "POVINF.DOC", "KNOWNBUGS.DOC"):
            commands += _fetch(
                "https://www.povray.org/ftp/pub/povray/Old-Versions/Official-2.2/" + filename,
                "povray-2.2/" + filename,
                revision="Official-2.2",
            )
    elif kind == "caffe":
        commands += _preserve_nproc_wrapper(4)
        commands += _git_mirror(
            "caffe",
            "https://github.com/BVLC/caffe.git",
            "eeebdab16155d34ff8f5f42137da7df4d1c7eab0",
            ref="refs/tags/1.0",
            aliases=("https://github.com/BVLC/caffe",),
        )
        commands += _apt_cache(
            [
                "build-essential",
                "cmake",
                "libprotobuf-dev",
                "protobuf-compiler",
                "libleveldb-dev",
                "libsnappy-dev",
                "libopencv-dev",
                "libhdf5-dev",
                "libboost-all-dev",
                "libgflags-dev",
                "libgoogle-glog-dev",
                "liblmdb-dev",
                "libopenblas-dev",
            ]
        )
        commands += _fetch(
            "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
            "cifar10/cifar-10-binary.tar.gz",
            revision="cifar-10-original-binary-distribution",
            md5="c32a1d4ab5d03f1284b67883e8d87530",
            aliases=(
                "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz",
            ),
        )
    elif kind == "compcert":
        commands += _preserve_nproc_wrapper(2)
        commands += _git_mirror(
            "compcert",
            "https://github.com/AbsInt/CompCert.git",
            "44d67d81b2a9ed5731d2bbfdf56d591ff3046ab5",
            ref="refs/tags/v3.13.1",
        )
        commands += _fetch(
            "https://github.com/AbsInt/CompCert/archive/refs/tags/v3.13.1.tar.gz",
            "compcert/v3.13.1.tar.gz",
            revision="44d67d81b2a9ed5731d2bbfdf56d591ff3046ab5",
        )
        commands += _apt_cache(["build-essential", "ocaml", "ocaml-findlib", "coq", "menhir", "libmenhir-ocaml-dev"])
    elif kind == "tokens":
        commands += _python_cache(["datasets==3.6.0", "transformers==4.56.0"])
        commands += _hf_snapshots(
            [
                {
                    "repo_id": "ryanmarten/OpenThoughts-1k-sample",
                    "repo_type": "dataset",
                    "revision": "a82400884621626d41bef89b7604f8054e7e00e0",
                },
                {
                    "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                    "ignore_patterns": ["model.safetensors"],
                },
            ],
            dataset=True,
        )
    elif kind == "distilbert":
        commands += _hf_snapshots(
            [
                {
                    "repo_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                    "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
                    "aliases": ["distilbert-base-uncased-finetuned-sst-2-english"],
                }
            ]
        )
    elif kind == "bge":
        commands += _hf_snapshots(
            [{"repo_id": "BAAI/bge-small-zh-v1.5", "revision": "7999e1d3359715c523056ef9478215996d62a620"}]
        )
    elif kind == "mteb_results":
        commands += _mteb_results()
    elif kind == "rstan":
        commands += _rstan_cache()
    elif kind == "pystan":
        commands += _apt_cache(["build-essential", "python3-dev"])
        commands += _python_cache(["pystan==3.10.0"])
    elif kind == "grpc":
        commands += _python_cache(["grpcio==1.73.0", "grpcio-tools==1.73.0"])
    elif kind == "pypi":
        commands += _python_cache(
            ["build==1.3.0", "packaging==25.0", "pypiserver==2.4.0", "setuptools==75.6.0", "wheel==0.45.1"]
        )
    elif kind == "reshard":
        commands += _python_cache(["datasets==3.6.0", "tqdm==4.67.1"])
    elif kind == "r":
        commands += _apt_cache(["r-base", "r-base-dev"])
    elif kind == "nginx":
        commands += _apt_cache(["nginx"])
    elif kind == "git_server":
        commands += _apt_cache(["git", "openssh-client", "openssh-server"])
    elif kind == "windows":
        commands += _apt_cache(["qemu-system-x86"])
        commands += _fetch(
            "https://download.qemu.org/qemu-5.2.0.tar.xz",
            "qemu/qemu-5.2.0.tar.xz",
            revision="v5.2.0",
        )
        commands += [
            "apt-get install -y --no-install-recommends util-linux",
            # Original www-data/adm IDs survive fakeroot chown and prevent
            # writable-tmpfs copy-up. New inodes drop that inherited metadata.
            WINDOWS_LOG_RESET,
            _write(f"{ROOT}/windows-startup.sh", WINDOWS_STARTUP),
            _write(f"{ROOT}/resource-env.sh", f"export BASH_ENV={ROOT}/windows-startup.sh", append=True),
        ]
    elif kind == "fasttext":
        commands += _git_mirror(
            "fasttext",
            "https://github.com/facebookresearch/fastText.git",
            "1142dc4c4ecbc19cc16eee5cdd28472e689267e6",
            aliases=("https://github.com/facebookresearch/fastText",),
        )
        commands += _apt_cache(["build-essential"])
    elif kind == "mobilesam":
        revision = "34bbbfdface3c18e5221aa7de6032d7220c6c6a1"
        commands += _git_mirror(
            "mobilesam",
            "https://github.com/ChaoningZhang/MobileSAM.git",
            revision,
            aliases=("https://github.com/ChaoningZhang/MobileSAM",),
        )
        commands += _fetch(
            f"https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/{revision}/weights/mobile_sam.pt",
            "mobilesam/mobile_sam.pt",
            revision=revision,
            aliases=("https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",),
        )
        commands.append(
            f'test "$(git hash-object {ROOT}/resources/mobilesam/mobile_sam.pt)" = "$(git --git-dir={ROOT}/resources/mobilesam.git rev-parse {revision}:weights/mobile_sam.pt)"'
        )
    else:
        raise ValueError(f"Unaudited resource recipe {kind!r} for {task_id}")
    return commands

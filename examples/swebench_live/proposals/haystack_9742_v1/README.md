# Proposed Haystack offline environment

The frozen Haystack image contains cached model assets. Its grading command does
not enable Hugging Face offline mode. Nine preservation tests therefore attempt
an `additional_chat_templates` metadata request from a container without network access.

This proposal prefixes the original grading command with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. The package APIs must use cached files and fail when
required files are absent. The task source, reference patch, test patch, test bodies,
two required bug checks, and 3,326 preservation checks stay unchanged.

The saved model patch also lacks `pipeline_outputs`. That is a separate task failure.
This proposal does not modify model patches or relabel any historical score.

The builder emits a separate record and runtime contract. The normal adapter selects
the environment only when this exact proposed record is present. It reads that record
after the solver stops and its candidate is sealed. The existing Apptainer grader
installs the capsule before applying the candidate. Its constructor and reset behavior
stay unchanged. The original solver image receives no new dependencies or test artifacts.
Earlier checkpoints cannot resume with the new evaluator digest.
Native validation must compare the unchanged base,
the supplied reference, and the saved candidate using the same complete tests.

Two additional, unscored tests require the upstream spaCy e2e dependencies.
`environment.json` pins the 25 missing packages and their Linux/Python 3.9 wheels.
It retains all 195 top-level packages from the original image. No existing package
changes version. The extra follows the frozen Haystack `pyproject.toml`: spaCy 3.8,
`en_core_web_trf` 3.8.0, and the Python 3.9 constraint `blis<1.2.1`.

`install_dependencies.py` checks the original inventory and every wheel hash before
installation. It uses local wheels without an index or dependency resolution.
It rejects an unexpected resulting inventory. Run it only inside a fresh disposable
copy of the pinned original image. Verify the SIF hash before unpacking that image.
The proposal binds this dependency manifest to its evaluator digest.

```sh
uv run python -m examples.swebench_live.proposals.haystack_9742_v1.prepare \
  --record /path/to/frozen/evaluator-record.json \
  --output /path/to/new/proposed-packet
```

Local command-dispatch tests use a labeled fake Hatch executable. They prove
environment and argument propagation, not model loading or native task correctness.
The native controls must load the actual frozen models and keep the complete scoring rules.

Download `requirements.txt` with hash checking and binary-only selection for Linux
x86_64 and CPython 3.9. Keep the resulting wheels outside the source repository.
The native installer takes `--wheelhouse /path/to/wheels --receipt /path/to/new/receipt.json`.
Use the native interpreter recorded in `environment.json` through `uv`.

Stage the capsule under a separate asset root. This command copies only the dependency
manifest and verified wheels. It does not copy the record, reference patch, or test results:

```sh
uv run python -m examples.swebench_live.proposals.haystack_9742_v1.deploy \
  --wheelhouse /path/to/verified/wheels \
  --isolated-assets /path/to/new/isolated-assets
```

The adapter requires `evaluation-environments/<manifest-sha256>` in its asset root.
Keep the original harness runtime and image identities when preparing that separate root.
The capsule and proposed record require explicit adoption before a new paid run.

The full native gate uses a separate sandbox for the original source base, supplied
reference patch, and saved model patch. It checks existing package file hashes,
cached model files, all required test IDs, and the complete official results.
Keep reference patches, test results, and diagnostic files out of future solver images.
The builder alone does not prove that this gate passes or authorize adoption.
Current Apptainer 1.5.5 differs from historical 1.5.0. Record that transport difference.

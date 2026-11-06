# SWE solver repository isolation

The solver receives a fresh, shallow Git repository for each native trial.
Its object database contains only the frozen base commit and that commit's
tree and blobs. The original commit ID stays unchanged. Parent commit objects,
other branches, tags, reflogs, replacement refs, grafts, alternates, and image
Git configuration are absent.

The protocol is `swe-base-commit-and-tree-only-v1`. It removes historical Git
context as well as future objects. This changes the solver's available evidence.
It is not lossless compression or equivalent to the previous solver protocol.
The runtime identity records the protocol and host Git identity. Existing run
contracts therefore reject resume under this implementation. Use a separately
approved attempt and a new output directory.

The adapter passes only the frozen evaluator record's `base_commit` to the
solver deployment. It does not expose solution patches or required test data.
Before an Apptainer instance starts, the host verifies the image checkout's HEAD
and every tracked file's raw bytes and Git mode against that exact commit.
It builds a new object database outside the container filesystem, verifies its
complete object allowlist, then removes the original `.git` directory.
Temporary copies of the original objects are removed before startup.

The new index supports ordinary edits, `git add -A`, and `git diff --cached`.
Tracked files, executable flags, symlink contents, immutable images, and grader
containers stay unchanged. Each solver trial records its isolation receipt in
`solver-repository-isolation.json` before the SWE-ReX worker starts.

Isolation fails before solver inference on a base mismatch, dirty tracked bytes,
unsafe paths, Git storage symlinks or hardlinks, nested Git stores, or missing
objects. Gitlinks and shared worktrees are unsupported and fail closed. Original
alternates are never followed; repositories that need them for the base objects
also fail closed. Repository and global Git configuration do not control the
isolation commands.

Offline tests use real repositories with packed future commits, annotated tags,
replacement refs, grafts, and alternates. They verify exact base content,
submission diffs, path and permission failures, startup ordering, and resume
rejection. They make no model calls and run no benchmark trial.

This repair removes the confirmed `/testbed/.git` exposure. It does not prove
that every image lacks independent copies of hidden content elsewhere. The
saved Haystack image remains unchanged. Native Apptainer execution of this new
protocol requires separate deployment and trial approval.

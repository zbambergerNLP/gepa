"""Pin the published AppWorld engine, data, and ReAct execution protocol."""

APPWORLD_VERSION = "0.1.3.post1"
APPWORLD_REVISION = "66ad8099e12188ece0d3fe45e661dbc01880813b"
PYTHON_VERSION = "3.11.14"
DATA_VERSION = "0.1.0"
DATA_URL = "https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.1.0.bundle"
DATA_BUNDLE_SHA256 = "fd9f9608c2ec71ed0ac25c3633a738b9129a318a129e31230425b9188e508250"
OFFICIAL_SPLITS = ("train", "dev", "test_normal", "test_challenge")
DEFAULT_MAX_STEPS = 100
ENVIRONMENT_SEED = 100
CODE_TIMEOUT_SECONDS = 100
RPC_TIMEOUT_SECONDS = 300
HARNESS_VERSION = "appworld-react-gepa-v1"
COMPONENT = "system_prompt"

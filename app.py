from pathlib import Path
import sys

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_ai.api.app import create_app  # noqa: E402
from legal_ai.config import get_settings  # noqa: E402
from legal_ai.logging_utils import configure_logging  # noqa: E402


configure_logging(get_settings().log_level)
app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


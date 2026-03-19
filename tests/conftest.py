"""Pytest configuration for LESSON-Bench test suite.

Adds the project root to sys.path so all lesson.* imports resolve correctly
without needing an editable install.
"""

import sys
from pathlib import Path

# Insert project root so `import lesson` works regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalized_requirement_names() -> set[str]:
    names = set()

    for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].lower())

    return names


def _normalized_project_dependencies() -> set[str]:
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {
        dependency.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].lower()
        for dependency in data["project"]["dependencies"]
    }


class DependencyDeclarationTest(unittest.TestCase):
    def test_core_runtime_dependencies_are_declared(self):
        expected = {"docling", "ollama"}

        self.assertLessEqual(expected, _normalized_requirement_names())
        self.assertLessEqual(expected, _normalized_project_dependencies())


if __name__ == "__main__":
    unittest.main()

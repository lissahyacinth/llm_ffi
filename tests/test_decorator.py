import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from llm_ffi import decorator
from llm_ffi.llm_connector import LLMInterface


def implementation(offset: int) -> str:
    return (
        "def add(value: int, *, increment: int = 1) -> int:\n"
        f"    return value + increment + {offset}\n"
    )


class GenerationModeTests(TestCase):
    def setUp(self) -> None:
        self.enterContext(
            patch.dict(
                os.environ,
                {"LLM_FFI_DANGER_MODE": "", "LLM_FFI_MORE_DANGER_MODE": ""},
            )
        )
        self.llm = Mock(spec=LLMInterface)
        self.enterContext(
            patch.object(decorator._LLMClientSingleton, "get", return_value=self.llm)
        )
        directory = Path(self.enterContext(TemporaryDirectory()))
        module_path = directory / "subject.py"
        module_path.write_text(
            "from llm_ffi import llm_impl\n\n"
            "@llm_impl\n"
            "def add(value: int, *, increment: int = 1) -> int:\n"
            '    """Add increment to value."""\n'
            "    raise NotImplementedError\n"
        )
        # Unique module names isolate runtime cache entries between tests.
        module_name = f"subject_{directory.name}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.add = module.add
        self.generated_path = directory / "subject_llm_generated/add.llm_generated.py"

    def test_more_danger_generates_and_executes_on_every_call(self) -> None:
        os.environ["LLM_FFI_MORE_DANGER_MODE"] = "true"
        self.llm.infer.side_effect = [
            implementation(10),
            implementation(20),
            implementation(30),
        ]

        self.assertEqual(self.add(1, increment=2), 13)
        self.assertEqual(self.add(1, increment=2), 23)

        self.assertEqual(self.llm.infer.call_count, 2)
        self.assertFalse(self.generated_path.parent.exists())

        os.environ["LLM_FFI_MORE_DANGER_MODE"] = "false"
        os.environ["LLM_FFI_DANGER_MODE"] = "true"
        self.assertEqual(self.add(1, increment=2), 33)
        self.assertEqual(self.add(2, increment=3), 35)
        self.assertEqual(self.llm.infer.call_count, 3)

    def test_more_danger_overrides_saved_code_and_warm_caches(self) -> None:
        os.environ["LLM_FFI_DANGER_MODE"] = "true"
        self.llm.infer.side_effect = [
            implementation(10),
            implementation(20),
            implementation(30),
        ]
        self.assertEqual(self.add(1), 12)
        self.generated_path.parent.mkdir()
        saved_code = implementation(100)
        self.generated_path.write_text(saved_code)
        self.assertEqual(self.add(1), 102)
        os.environ["LLM_FFI_MORE_DANGER_MODE"] = "true"
        self.assertEqual(self.add(1), 22)
        self.assertEqual(self.add(1), 32)

        self.assertEqual(self.llm.infer.call_count, 3)
        self.assertEqual(self.generated_path.read_text(), saved_code)
        os.environ["LLM_FFI_MORE_DANGER_MODE"] = "false"
        self.generated_path.write_text(implementation(200))
        self.assertEqual(self.add(1), 102)
        self.generated_path.unlink()
        self.assertEqual(self.add(1), 12)
        self.assertEqual(self.llm.infer.call_count, 3)

    def test_danger_mode_still_reuses_generated_implementation(self) -> None:
        os.environ["LLM_FFI_DANGER_MODE"] = "true"
        self.llm.infer.return_value = implementation(10)

        self.assertEqual(self.add(1), 12)
        self.assertEqual(self.add(2, increment=3), 15)
        self.llm.infer.assert_called_once()

    def test_normal_mode_requires_saved_code(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Generated code not found"):
            self.add(1)
        self.llm.infer.assert_not_called()

    def test_more_danger_propagates_failures_and_generates_again_next_call(
        self,
    ) -> None:
        os.environ["LLM_FFI_MORE_DANGER_MODE"] = "true"
        failures = [
            (RuntimeError("Inference failed"), RuntimeError),
            ("def add(:", SyntaxError),
            ("def add(value):\n    raise ValueError('No profit')\n", ValueError),
        ]
        for response, error in failures:
            with self.subTest(error=error):
                self.llm.infer.reset_mock()
                self.llm.infer.side_effect = [
                    implementation(10),
                    response,
                    implementation(20),
                ]
                self.assertEqual(self.add(1), 12)
                with self.assertRaises(error):
                    self.add(1)
                self.assertEqual(self.add(1), 22)
                self.assertEqual(self.llm.infer.call_count, 3)

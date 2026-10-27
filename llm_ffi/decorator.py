import ast
import importlib.util
import inspect
import os
import re
import threading
from collections.abc import Callable
from functools import wraps
from pathlib import Path

from llm_ffi.function import FunctionSpec
from llm_ffi.llm_connector import LLMExternalInference, LLMInterface
from llm_ffi.utils import compute_hash

_IMPL_CACHE: dict[str, Callable] = {}
_RUNTIME_CACHE: dict[str, Callable] = {}


def danger_mode_enabled() -> bool:
    """Check if runtime generation is enabled via environment variable"""
    return os.getenv("LLM_FFI_DANGER_MODE", "").lower() in ("true", "1", "yes")


def more_danger_mode_enabled() -> bool:
    """Check if every call should generate and execute a fresh implementation"""
    return os.getenv("LLM_FFI_MORE_DANGER_MODE", "").lower() in ("true", "1", "yes")


class _LLMClientSingleton:
    """Thread-safe singleton for LLM client used in runtime generation"""

    _instance: LLMInterface | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> LLMInterface:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    model = os.getenv("LLM_FFI_MODEL", "claude-sonnet-4-5-20250929")
                    cls._instance = LLMExternalInference.from_config(model)
        return cls._instance


def extract_function_spec(func: Callable) -> FunctionSpec:
    """Extract spec from function at runtime"""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    source = inspect.getsource(func)
    spec_text = f"{func.__name__}{sig}\n{doc}"

    return FunctionSpec(
        name=func.__name__,
        source=source,
        spec_hash=compute_hash(spec_text),
        code_hash="",
    )


def compile_and_extract_function(code: str, func_name: str) -> Callable:
    """Compile generated code and extract the function"""
    tree = ast.parse(code)
    compiled = compile(tree, "<llm-generated>", "exec")
    namespace = {}
    exec(compiled, namespace)  # noqa: S102 - Executing generated code is the FFI.
    if func_name not in namespace:
        raise RuntimeError(f"Generated code missing function '{func_name}'")
    return namespace[func_name]


def _generate_function(spec: FunctionSpec) -> Callable:
    """Generate and compile a fresh implementation"""
    from llm_ffi.generate import generate_implementation

    llm = _LLMClientSingleton.get()
    code = generate_implementation(spec, llm)
    # Strip markdown code fences if present
    code = re.sub(r"^```python\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return compile_and_extract_function(code, spec.name)


def _runtime_generate(func: Callable, *args, **kwargs):
    """Generate implementation at runtime and cache it"""
    spec = extract_function_spec(func)
    cache_key = f"{func.__module__}.{func.__name__}:{spec.spec_hash}"

    if cache_key not in _RUNTIME_CACHE:
        _RUNTIME_CACHE[cache_key] = _generate_function(spec)

    return _RUNTIME_CACHE[cache_key](*args, **kwargs)


def llm_impl(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if more_danger_mode_enabled():
            impl = _generate_function(extract_function_spec(func))
            return impl(*args, **kwargs)

        module_path = Path(inspect.getfile(func))
        generated_path = (
            module_path.parent
            / f"{module_path.stem}_llm_generated"
            / f"{func.__name__}.llm_generated.py"
        )

        if not generated_path.exists():
            if danger_mode_enabled():
                return _runtime_generate(func, *args, **kwargs)
            else:
                raise RuntimeError(
                    f"Generated code not found: {generated_path}\n"
                    f"Run: llm-ffi generate {module_path}"
                )

        # File exists - use it
        cache_key = str(generated_path)
        if cache_key not in _IMPL_CACHE:
            spec = importlib.util.spec_from_file_location("_generated", generated_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _IMPL_CACHE[cache_key] = getattr(module, func.__name__)

        return _IMPL_CACHE[cache_key](*args, **kwargs)

    return wrapper

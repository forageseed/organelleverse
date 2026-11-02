"""The Rust crate is found from ORGANELLEVERSE_NATIVE_DIR or the checkout's build directory."""

from pathlib import Path

from organelleverse import accel


def test_the_configured_directory_is_searched_first(monkeypatch, tmp_path):
    monkeypatch.setenv(accel.NATIVE_DIR_ENV, str(tmp_path))
    directories = list(accel._native_directories())
    assert directories[0] == tmp_path
    assert directories[-1].parts[-3:] == ("rust", "target", "release")


def test_without_the_variable_only_the_checkout_build_is_searched(monkeypatch):
    monkeypatch.delenv(accel.NATIVE_DIR_ENV, raising=False)
    directories = list(accel._native_directories())
    assert len(directories) == 1 and directories[0].name == "release"


def test_a_missing_or_empty_directory_loads_nothing(tmp_path):
    assert accel._load_from_directory(tmp_path / "absent") is None
    assert accel._load_from_directory(tmp_path) is None


def test_a_file_that_is_not_a_python_extension_loads_nothing(tmp_path):
    for name in (
        "organelleverse_rust.cpython-313-x86_64-linux-gnu.so",
        "liborganelleverse_rust.so",
        "organelleverse_rust.d",
    ):
        Path(tmp_path / name).write_bytes(b"not an ELF object")
    assert accel._load_from_directory(tmp_path) is None

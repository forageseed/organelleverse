"""LOSAT is found from a worktree, the BLAST+ fallback warns, preflight has a longer default."""

import warnings

import pytest

from organelleverse.annotation import execution
from organelleverse.annotation.backends import plastome as backend_module
from organelleverse.annotation.plastome import tools


def test_preflight_timeout_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv(execution.PREFLIGHT_TIMEOUT_ENV, raising=False)
    assert execution._preflight_timeout(None) == 120
    assert execution._preflight_timeout(7) == 7
    monkeypatch.setenv(execution.PREFLIGHT_TIMEOUT_ENV, "300")
    assert execution._preflight_timeout(None) == 300
    monkeypatch.setenv(execution.PREFLIGHT_TIMEOUT_ENV, "soon")
    assert execution._preflight_timeout(None) == 120


# The checkout comes from organelleverse._layout.checkout_root() (layout-aware
# since the flat src/ move), no longer from counting parents of tools.__file__.


def test_worktree_also_searches_the_repository_root(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    here = root / ".worktrees" / "feature"
    here.mkdir(parents=True)
    monkeypatch.setattr(tools, "checkout_root", lambda: here)
    assert tools._checkout_roots() == (here, root)


def test_plain_checkout_searches_only_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "checkout_root", lambda: tmp_path)
    assert tools._checkout_roots() == (tmp_path,)


@pytest.mark.parametrize("token", ["none", "OFF", "ncbi", "0"])
def test_opt_out_tokens_are_recognised(monkeypatch, token):
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", token)
    assert tools.losat_opted_out()


def _fallback(monkeypatch):
    monkeypatch.setattr(backend_module, "_resolve_losat", lambda _p: None)
    monkeypatch.setattr(
        backend_module,
        "resolve_required_tools",
        lambda names, runner=None, paths=None: tuple(
            type("T", (), {"name": n, "path": f"/bin/{n}", "version": "2.16"})() for n in names
        ),
    )


def test_blast_fallback_warns_unless_opted_out(monkeypatch):
    _fallback(monkeypatch)
    monkeypatch.delenv("ORG_VERSE_LOSAT_BIN", raising=False)
    with pytest.warns(RuntimeWarning, match="LOSAT was not found"):
        paths, _ = backend_module._resolve_search_tools(runner=None)
    assert set(paths) == {"blastn", "makeblastdb", "tblastn"}

    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "none")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend_module._resolve_search_tools(runner=None)

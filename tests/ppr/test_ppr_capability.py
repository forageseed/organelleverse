import inspect

from organelleverse.capabilities.discovery import discover_capability_candidates
from organelleverse.ppr.predict import predict_binding_sites


def test_ppr_capability_is_native_and_declares_path_inputs() -> None:
    index = discover_capability_candidates()
    entry = index.describe("ppr.predict_binding_sites")

    assert entry.bundle.contract.callable_locator == "organelleverse.ppr.predict:predict_binding_sites"
    # Native core capability: the implementation signature is the parameter source.
    assert set(inspect.signature(predict_binding_sites).parameters) == {
        "repeat_tsv",
        "transcript_fasta",
        "organelle",
        "extended_code_map_json",
    }
    assert {parameter.name for parameter in entry.bundle.contract.binding.parameters} == {
        "repeat_tsv",
        "transcript_fasta",
        "organelle",
        "extended_code_map_json",
    }

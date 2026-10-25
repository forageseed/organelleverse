"""Native-only arguments excluded from the Agent JSON parameter schema.

Native scientific data retain their Python types and validation. Every new
exclusion needs a reason and is checked by test_public_validation.py.
"""

from types import MappingProxyType

PYTHON_ONLY_PARAMETERS = MappingProxyType(
    {
        "coevolution.phylogenomics": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "coevolution.rate_covariation": {
            "compartment_map": "Native structured object; requires a "
            "designed closed JSON/artifact codec"
        },
        "coevolution.run_coevolution": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "coevolution.run_orthofinder": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "comparative.compute_genome_identity": {
            "table_output": "Host executable/location or explicit Python publication destination"
        },
        "fetch.accession_list": {
            "transport": "Native Python injection or rich filter object; passed unchanged"
        },
        "fetch.dedup_genomes": {
            "records": "Native records sequence; JSON uses the canonical data core input"
        },
        "fetch.entrez_query": {
            "filters": "Native Python injection or rich filter object; passed unchanged",
            "transport": "Native Python injection or rich filter object; passed unchanged",
        },
        "fetch.gene_records": {
            "transport": "Native Python injection or rich filter object; passed unchanged"
        },
        "fetch.genome_size_candidates": {
            "transport": "Native Python injection or rich filter object; passed unchanged"
        },
        "fetch.ngdc_gwh": {
            "transport": "Native Python injection or rich filter object; passed unchanged"
        },
        "fetch.nuclear_assembly": {
            "transport": "Native Python injection or rich filter object; passed unchanged"
        },
        "fetch.refseq_snapshot": {
            "transport": "Native Python injection or rich filter object; passed unchanged"
        },
        "hgt.detect_hgt": {
            "blastn_path": "Host executable/location or explicit Python publication destination",
            "makeblastdb_path": "Host executable/location or explicit Python publication "
            "destination",
            "minimap2_path": "Host executable/location or explicit Python publication destination",
        },
        "localization.predict": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "morphology.correct": {
            "corrected_at": "Native structured object; requires a designed closed "
            "JSON/artifact codec"
        },
        "morphology.overlay": {
            "palette": "Native structured object; requires a designed closed JSON/artifact codec"
        },
        "morphology.segment": {
            "executor": "Python injection or rich filter object; passed unchanged",
            "orgseg_root": "Host executable/location or explicit Python publication destination",
        },
        "morphology.train": {
            "executor": "Python injection or rich filter object; passed unchanged",
            "orgseg_root": "Host executable/location or explicit Python publication destination",
        },
        "morphology.verify": {
            "client": "Python injection or rich filter object; passed unchanged",
            "store_path": "Host executable/location or explicit Python publication destination",
        },
        "phylogeny.build_tree": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "phylogeny.run_build_tree": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "phylogeny.run_trim_alignment": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "phylogeny.trim_alignment": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "population.call_variants": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "population.cytonuclear_gwas": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "population.prepare_gemma_input": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "selection.align_protein": {
            "executor": "Python injection or rich filter object; passed unchanged"
        },
        "selection.batch_codeml": {
            "codeml_path": "Host executable/location or explicit Python publication destination"
        },
        "selection.check_hyphy": {
            "hyphy_path": "Host executable/location or explicit Python publication destination"
        },
        "selection.hyphy_absrel": {
            "hyphy_path": "Host executable/location or explicit Python publication destination"
        },
        "selection.hyphy_busted": {
            "hyphy_path": "Host executable/location or explicit Python publication destination"
        },
        "selection.hyphy_fel": {
            "hyphy_path": "Host executable/location or explicit Python publication destination"
        },
        "selection.hyphy_meme": {
            "hyphy_path": "Host executable/location or explicit Python publication destination"
        },
        "selection.hyphy_relax": {
            "hyphy_path": "Host executable/location or explicit Python publication destination"
        },
        "selection.kaks_calculator": {
            "kaks_path": "Host executable/location or explicit Python publication destination"
        },
        "selection.run_kaks_calculator_workflow": {
            "kaks_path": "Host executable/location or explicit Python publication destination"
        },
        "structure.compute_repeats": {
            "unit_thresholds": "Native structured object; requires a designed "
            "closed JSON/artifact codec"
        },
        "structure.repeats": {
            "unit_thresholds": "Native structured object; requires a designed closed "
            "JSON/artifact codec"
        },
        "transfer.detect_transfers_blast": {
            "blastn_path": "Host executable/location or explicit Python publication destination",
            "makeblastdb_path": "Host executable/location or explicit "
            "Python publication destination",
        },
        "transfer.detect_transfers_evidence": {
            "blastn_path": "Host executable/location or explicit Python publication destination",
            "makeblastdb_path": "Host executable/location or explicit "
            "Python publication destination",
            "minimap2_path": "Host executable/location or explicit Python publication destination",
        },
        "transfer.validate_transfers_longread": {
            "minimap2_path": "Host executable/location or explicit Python publication destination"
        },
        "visualization.ideogram": {
            "label": "Native structured object; requires a designed closed JSON/artifact codec",
            "overlaid": "Native structured object; requires a designed closed JSON/artifact codec",
            "profile": "Native structured object; requires a designed closed JSON/artifact codec",
        },
        "visualization.nuclear_transfer_ideogram": {
            "gene_hits": "Native structured object; requires a designed closed JSON/artifact codec",
            "plastid_fragments": "Native structured object; "
            "requires a designed closed "
            "JSON/artifact codec",
        },
        "visualization.plot_assembly_graph": {
            "minimap2": "Host executable/location or explicit Python publication destination"
        },
        "visualization.plot_collinearity": {
            "kwargs": "Native variadic escape hatch; no closed JSON field"
        },
        "visualization.plot_fst_scan": {
            "windows": "Native structured object; requires a designed closed JSON/artifact codec"
        },
        "visualization.plot_gbdraw": {
            "_extra_flags": "Private Python routing argument; not an Agent scientific option",
            "_op": "Private Python routing argument; not an Agent scientific option",
            "executable": "Host executable/location or explicit Python publication destination",
            "runner": "Python injection or rich filter object; passed unchanged",
            "session": "Host executable/location or explicit Python publication destination",
        },
        "visualization.plot_gene_structure": {
            "kwargs": "Native variadic escape hatch; no closed JSON field"
        },
        "visualization.plot_gfa_graph": {
            "executable": "Host executable/location or explicit Python publication destination",
            "runner": "Python injection or rich filter object; passed unchanged",
        },
        "visualization.plot_ogdraw_map": {
            "kwargs": "Native variadic escape hatch; no closed JSON field"
        },
        "visualization.plot_pan_circular": {
            "bandage_executable": "Host executable/location or explicit "
            "Python publication destination",
            "bandage_runner": "Python injection or rich filter object; passed unchanged",
        },
        "visualization.plot_presence_tracks": {
            "coverage": "Native structured object; requires a designed closed JSON/artifact codec"
        },
        "visualization.plot_sv_frequency": {
            "coverage": "Native structured object; requires a designed closed JSON/artifact codec"
        },
        "visualization.plot_sv_schematic": {
            "schematic": "Native structured object; requires a designed closed JSON/artifact codec"
        },
        "visualization.plot_track_density": {
            "regions": "Native structured object; requires a designed closed JSON/artifact codec"
        },
        "visualization.plot_transfer_tracks": {
            "plastid_candidates": "Native structured object; requires "
            "a designed closed JSON/artifact "
            "codec"
        },
    }
)

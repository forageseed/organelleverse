"""NOVOPlasty 4.3.5 configuration, candidate collection and seed provenance."""

from __future__ import annotations

import json
import re
import shutil
from importlib.resources import files
from pathlib import Path

from Bio import SeqIO

from organelleverse.assembly.backends.base import (
    AdapterContext,
    AssemblyCommand,
    BackendOutput,
    NormalizedAssemblyOutputs,
    RawAssemblyOutputs,
)
from organelleverse.assembly.data_contract import validate_novoplasty_assembly_data
from organelleverse.assembly.normalization import normalize_fasta
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError


def _file(role: str, path: Path, fmt: str = "fasta") -> BackendOutput:
    return BackendOutput(
        role=role,
        path=path,
        format=fmt,
        media_type="text/x-fasta" if fmt == "fasta" else "application/json",
    )


def _yes(value: object) -> str:
    return "yes" if value else "no"


class NovoplastyAdapter:
    """Preserve all candidates without treating option ordering as a ranking.

    Only Circularized_assembly files establish circularity. Option files are
    alternative joins with unresolved topology; Contigs files are linear
    fragments. No sequence concatenation or inferred circularization is used.
    """

    backend_id = "novoplasty"

    def preflight(self, context: AdapterContext) -> None:
        validate_novoplasty_assembly_data(context.request.data)
        if (
            context.route.selected_backend != self.backend_id
            or context.environment.backend_id != self.backend_id
        ):
            raise OrganelleInputError(
                code="assembly.unsupported_data_profile",
                message="NOVOPlasty requires its verified environment and route",
            )
        provided = context.request.backend_parameters
        if provided is not None and provided.backend != self.backend_id:
            raise OrganelleInputError(
                code="assembly.unsupported_data_profile",
                message="backend_parameters must match NOVOPlasty",
            )

    def build_command(self, context: AdapterContext) -> AssemblyCommand:
        self.preflight(context)
        params = context.effective_backend_parameters
        root = context.workspace / "backend" / self.backend_id
        root.mkdir(parents=True, exist_ok=True)
        library = context.payload.short_libraries[0]
        auxiliary = context.payload.auxiliary

        def stage(role: str | None, name: str) -> str:
            if role is None:
                return ""
            source = Path(context.input_artifacts[role].uri)
            # Upstream parses one field per line. Local safe filenames also
            # avoid carrying user punctuation into its internal shell commands.
            target = root / (name + (".gz" if source.name.endswith(".gz") else ""))
            shutil.copyfile(source, target)
            return str(target)

        forward = stage(library.read1_artifact, "reads_1.fastq")
        reverse = stage(library.read2_artifact, "reads_2.fastq")
        seed_path = root / "seed.fasta"
        if auxiliary.seed_fasta_artifact is not None:
            seed = stage(auxiliary.seed_fasta_artifact, "seed.fasta")
            # Seed and reference input are plain FASTA, unlike reads.
            normalize_fasta(Path(seed), seed_path)
            seed_info = {
                "source": "user",
                "artifact_role": auxiliary.seed_fasta_artifact,
                "uri": context.input_artifacts[auxiliary.seed_fasta_artifact].uri,
            }
        else:
            resource = files("organelleverse.annotation").joinpath(
                "data/plastome/references/Arabidopsis_thaliana_chloroplast.gb"
            )
            with resource.open() as handle:
                record = SeqIO.read(handle, "genbank")
            feature = next(
                f
                for f in record.features
                if f.type == "CDS" and f.qualifiers.get("gene") == ["rbcL"]
            )
            seed_path.write_text(f">{record.id}_rbcL\n{feature.extract(record.seq)}\n")
            seed_info = {
                "source": "bundled",
                "resource": str(resource),
                "accession": record.id,
                "gene": "rbcL",
            }
        seed_info["records"] = [
            {"id": r.id, "length": len(r)} for r in SeqIO.parse(seed_path, "fasta")
        ]
        (root / "seed.json").write_text(json.dumps(seed_info, indent=2) + "\n")
        reference = stage(auxiliary.reference_fasta_artifact, "reference.fasta")
        chloroplast = stage(auxiliary.chloroplast_fasta_artifact, "chloroplast.fasta")
        interval = params["genome_range"]
        entries = [
            ("Project name", "organelleverse"),
            ("Type", params["type"]),
            ("Genome Range", f"{interval[0]}-{interval[1]}"),
            ("K-mer", params["kmer_size"]),
            ("Max memory", context.request.memory_gb or ""),
            ("Extended log", int(params["extended_log"])),
            ("Save assembled reads", _yes(params["save_assembled_reads"])),
            ("Seed Input", str(seed_path)),
            ("Extend seed directly", "no"),
            ("Reference sequence", reference),
            ("Variance detection", ""),
            ("Chloroplast sequence", chloroplast),
            ("Read Length", library.read_length),
            ("Insert size", library.insert_size),
            ("Platform", "illumina"),
            ("Single/Paired", "PE"),
            ("Combined reads", ""),
            ("Forward reads", forward),
            ("Reverse reads", reverse),
            ("Store Hash", ""),
            ("MAF", ""),
            ("HP exclude list", ""),
            ("PCR-free", ""),
            ("Insert size auto", _yes(params["insert_size_auto"])),
            ("Use Quality Scores", _yes(params["use_quality_scores"])),
            ("Reduce ambigious N's", ""),
            ("Output path", str(root) + "/"),
        ]
        config = root / "config.txt"
        config.write_text("\n".join(f"{key:<24} = {value}" for key, value in entries) + "\n")
        # Managed prefixes expose physical executable names; resolved providers
        # expose capability roles. Both are established environment contracts.
        primary_name = next(
            item.name
            for item in context.environment.executables
            if item.name in {"novoplasty", "NOVOPlasty4.3.5.pl"}
        )
        executable = context.environment.require_executable(primary_name)
        return AssemblyCommand(
            stable_argv=(
                "NOVOPlasty4.3.5.pl",
                "-c",
                "role://workspace/input",
            ),
            resolved_argv=(str(executable), "-c", str(config)),
        )

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs:
        root = context.workspace / "backend" / self.backend_id
        outputs = []
        for prefix in ("Circularized_assembly", "Option", "Contigs"):
            pattern = re.compile(rf"{prefix}_(\d+)_organelleverse\.fasta")
            candidates = [
                (int(match[1]), path)
                for path in root.glob(f"{prefix}_*_organelleverse.fasta")
                if (match := pattern.fullmatch(path.name))
            ]
            for number, path in sorted(candidates):
                if path.stat().st_size == 0:
                    raise OrganelleExecutionError(
                        code="assembly.output_incomplete",
                        message=f"NOVOPlasty produced an empty candidate: {path.name}",
                    )
                outputs.append(_file(f"{prefix}_{number}", path))
        if not outputs:
            raise OrganelleExecutionError(
                code="assembly.output_incomplete",
                message="NOVOPlasty produced no documented assembly candidates; inspect its log and seed",
            )
        return RawAssemblyOutputs(outputs=tuple(outputs), primary_sequence_role=outputs[0].role)

    def normalize(
        self, context: AdapterContext, raw: RawAssemblyOutputs, output_dir: Path
    ) -> NormalizedAssemblyOutputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        candidates = []
        alternates = []
        primary_stats = None
        for index, item in enumerate(raw.outputs):
            role = (
                "assembly_fasta"
                if item.role == raw.primary_sequence_role
                else f"alternate_fasta_{index}"
            )
            target = output_dir / f"{role}.fasta"
            stats = normalize_fasta(item.path, target)
            outputs.append(_file(role, target))
            if role == "assembly_fasta":
                primary_stats = stats
            else:
                alternates.append(role)
            topology = (
                "circular"
                if item.role.startswith("Circularized_assembly_")
                else ("linear" if item.role.startswith("Contigs_") else "unknown")
            )
            candidates.append(
                {
                    "role": role,
                    "source_file": item.path.name,
                    "topology": topology,
                    "records": stats.record_count,
                    "total_bases": stats.total_bases,
                }
            )
        root = context.workspace / "backend" / self.backend_id
        shutil.copyfile(root / "seed.fasta", output_dir / "novoplasty_seed.fasta")
        seed_info = json.loads((root / "seed.json").read_text())
        report = output_dir / "novoplasty_report.json"
        report.write_text(
            json.dumps(
                {
                    "seed": seed_info,
                    "candidates": candidates,
                    "primary_selection": "first circularized candidate, then option, then contigs; numeric file order, not biological ranking",
                },
                indent=2,
            )
            + "\n"
        )
        outputs.extend(
            (
                _file("novoplasty_seed", output_dir / "novoplasty_seed.fasta"),
                _file("novoplasty_report", report, "json"),
            )
        )
        assert primary_stats is not None
        return NormalizedAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="assembly_fasta",
            alternate_sequence_roles=tuple(alternates),
            record_count=primary_stats.record_count,
            total_bases=primary_stats.total_bases,
        )

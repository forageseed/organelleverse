# Python 与 agent 共用操作契约：迁移状态

2026-09-24。目标是让同一研究操作在人类 Python 调用、agent 工具调用和工作台中使用相同的操作 ID、参数 schema、校验、结构化结果及错误。输出文件由显式 writer 负责；科学计算本身返回结果。现有历史函数的签名和返回类型不能未经验证地批量改写。

## 已建立的共同入口

`organelleverse.capabilities.build_registry(index, trust_store=...)` 把剩余的发布操作与给定的已准入 capability 快照组装为一个 `OperationRegistry`。Python 调用方可使用其 `list()`、`describe()`、`parameter_schema()` 和 `invoke()`；agent 工具目录也调用同一个构建函数，并通过 `operations.adapters.json.invoke_json()` 获得结构化 JSON 响应。未准入、被拒绝或冲突的 bundle 不因构建 registry 而自动变成可调用操作。调用方须提供同一 `CapabilityIndex` 快照，才能比较两端是否看到同一集合。

```python
from organelleverse.capabilities import build_registry

registry = build_registry(admitted_index, trust_store=trust_store)
specs = registry.list()
schema = registry.parameter_schema("structure.compute_repeats")
result = registry.invoke("structure.compute_repeats", input=genome, parameters=params)
```

这里的 `admitted_index` 必须来自现有发现、验证和准入流程；代码片段不表示某个指定操作已在当前环境中准入或通过真实数据验证。

## 尚未统一的历史公开函数

本次共享入口只覆盖 `OperationRegistry` 中的操作。`tests/operations/test_output_boundary_public_surface.py` 目前冻结了 **19 个公开名称**的最终写出参数违规，其中别名共享函数对象：`coevolution` 2、`localization` 2、`morphology` 3、`pangenome` 1、`phylogeny` 3、`population` 3、`selection` 3、`visualization` 2。另有未注册的历史公开函数，其参数和返回值不一定符合 `CoreObject` 契约；不能把“有共享入口”等同于“所有 API 已统一”。

逐项迁移时，应先为原函数建立真实输入、输出和副作用对照测试；把纯计算结果与显式 writer 分开；为计算入口定义 capability contract、参数 schema、结果 codec、依赖与错误映射；让 Python 和 agent 调用相同的绑定；最后保留确有外部调用者需要的兼容入口，并在迁移清单中说明其退出条件。每项需要科学基准及默认测试通过，不能仅靠接口形状判断功能正确。现有冻结测试应随实际消除的违规同步缩小，不以重新分类或改名隐藏违规。

论文当前只可陈述“**已注册操作**共用契约”；“所有公开历史 API 已统一”仍是开发目标。


## Shared validation for ordinary Python calls

The 250 registered public scientific entry points use the same
`_invoke_checked_python` boundary as registry calls. Native input objects,
signatures and direct return types are preserved. JSON adapters retain their
explicit path, artifact and result codecs. Scientific values are validated
strictly from implementation annotations before execution; invalid arguments
raise `OrganelleParameterError`. Core input/output data contracts also apply.
`help(function)` identifies the validation boundary and points to
`ov.operations.parameter_schema(operation_id)`.

The bindings now include GC window size, tree method/bootstrap/seed/threads/model,
variant reference/method/ploidy, timeouts and plotting controls. GC windows are
positive integers; RNA editing thresholds lie in [0, 1]; fetch enum choices are
closed. Accession lists default to plastid. A null RefSeq release resolves and
records NCBI's current release; a supplied release must match the retained release.

Executors, transports and clients retain identity and are never JSON fields.
Native data structures, host executable locations and explicit Python output
destinations retain their Python interfaces. They are not silently serialized
or replaced by guessed values. The installed mapping
`organelleverse.operations.python_only.PYTHON_ONLY_PARAMETERS` records every
exception below. A regression test compares it with every public signature.

| Operation | Python-only argument | Reason |
|---|---|---|
| `coevolution.phylogenomics` | `executor` | Python injection or rich filter object; passed unchanged |
| `coevolution.rate_covariation` | `compartment_map` | Native structured object; requires a designed closed JSON/artifact codec |
| `coevolution.run_coevolution` | `executor` | Python injection or rich filter object; passed unchanged |
| `coevolution.run_orthofinder` | `executor` | Python injection or rich filter object; passed unchanged |
| `comparative.compute_genome_identity` | `table_output` | Host executable/location or explicit Python publication destination |
| `fetch.accession_list` | `transport` | Native Python injection or rich filter object; passed unchanged |
| `fetch.dedup_genomes` | `records` | Native records sequence; JSON uses the canonical data core input |
| `fetch.entrez_query` | `filters` | Native Python injection or rich filter object; passed unchanged |
| `fetch.entrez_query` | `transport` | Native Python injection or rich filter object; passed unchanged |
| `fetch.gene_records` | `transport` | Native Python injection or rich filter object; passed unchanged |
| `fetch.genome_size_candidates` | `transport` | Native Python injection or rich filter object; passed unchanged |
| `fetch.ngdc_gwh` | `transport` | Native Python injection or rich filter object; passed unchanged |
| `fetch.nuclear_assembly` | `transport` | Native Python injection or rich filter object; passed unchanged |
| `fetch.refseq_snapshot` | `transport` | Native Python injection or rich filter object; passed unchanged |
| `hgt.detect_hgt` | `minimap2_path` | Host executable/location or explicit Python publication destination |
| `hgt.detect_hgt` | `blastn_path` | Host executable/location or explicit Python publication destination |
| `hgt.detect_hgt` | `makeblastdb_path` | Host executable/location or explicit Python publication destination |
| `localization.predict` | `executor` | Python injection or rich filter object; passed unchanged |
| `morphology.correct` | `corrected_at` | Native structured object; requires a designed closed JSON/artifact codec |
| `morphology.overlay` | `palette` | Native structured object; requires a designed closed JSON/artifact codec |
| `morphology.segment` | `executor` | Python injection or rich filter object; passed unchanged |
| `morphology.segment` | `orgseg_root` | Host executable/location or explicit Python publication destination |
| `morphology.train` | `executor` | Python injection or rich filter object; passed unchanged |
| `morphology.train` | `orgseg_root` | Host executable/location or explicit Python publication destination |
| `morphology.verify` | `client` | Python injection or rich filter object; passed unchanged |
| `morphology.verify` | `store_path` | Host executable/location or explicit Python publication destination |
| `phylogeny.build_tree` | `executor` | Python injection or rich filter object; passed unchanged |
| `phylogeny.run_build_tree` | `executor` | Python injection or rich filter object; passed unchanged |
| `phylogeny.run_trim_alignment` | `executor` | Python injection or rich filter object; passed unchanged |
| `phylogeny.trim_alignment` | `executor` | Python injection or rich filter object; passed unchanged |
| `population.call_variants` | `executor` | Python injection or rich filter object; passed unchanged |
| `population.cytonuclear_gwas` | `executor` | Python injection or rich filter object; passed unchanged |
| `population.prepare_gemma_input` | `executor` | Python injection or rich filter object; passed unchanged |
| `selection.align_protein` | `executor` | Python injection or rich filter object; passed unchanged |
| `selection.batch_codeml` | `codeml_path` | Host executable/location or explicit Python publication destination |
| `selection.check_hyphy` | `hyphy_path` | Host executable/location or explicit Python publication destination |
| `selection.hyphy_absrel` | `hyphy_path` | Host executable/location or explicit Python publication destination |
| `selection.hyphy_busted` | `hyphy_path` | Host executable/location or explicit Python publication destination |
| `selection.hyphy_fel` | `hyphy_path` | Host executable/location or explicit Python publication destination |
| `selection.hyphy_meme` | `hyphy_path` | Host executable/location or explicit Python publication destination |
| `selection.hyphy_relax` | `hyphy_path` | Host executable/location or explicit Python publication destination |
| `selection.kaks_calculator` | `kaks_path` | Host executable/location or explicit Python publication destination |
| `selection.run_kaks_calculator_workflow` | `kaks_path` | Host executable/location or explicit Python publication destination |
| `structure.compute_repeats` | `unit_thresholds` | Native structured object; requires a designed closed JSON/artifact codec |
| `structure.repeats` | `unit_thresholds` | Native structured object; requires a designed closed JSON/artifact codec |
| `transfer.detect_transfers_blast` | `blastn_path` | Host executable/location or explicit Python publication destination |
| `transfer.detect_transfers_blast` | `makeblastdb_path` | Host executable/location or explicit Python publication destination |
| `transfer.detect_transfers_evidence` | `blastn_path` | Host executable/location or explicit Python publication destination |
| `transfer.detect_transfers_evidence` | `makeblastdb_path` | Host executable/location or explicit Python publication destination |
| `transfer.detect_transfers_evidence` | `minimap2_path` | Host executable/location or explicit Python publication destination |
| `transfer.validate_transfers_longread` | `minimap2_path` | Host executable/location or explicit Python publication destination |
| `visualization.ideogram` | `overlaid` | Native structured object; requires a designed closed JSON/artifact codec |
| `visualization.ideogram` | `label` | Native structured object; requires a designed closed JSON/artifact codec |
| `visualization.ideogram` | `profile` | Native structured object; requires a designed closed JSON/artifact codec |
| `visualization.nuclear_transfer_ideogram` | `gene_hits` | Native structured object; requires a designed closed JSON/artifact codec |
| `visualization.nuclear_transfer_ideogram` | `plastid_fragments` | Native structured object; requires a designed closed JSON/artifact codec |
| `visualization.plot_assembly_graph` | `minimap2` | Host executable/location or explicit Python publication destination |
| `visualization.plot_collinearity` | `kwargs` | Native variadic escape hatch; no closed JSON field |
| `visualization.plot_fst_scan` | `windows` | Native structured object; requires a designed closed JSON/artifact codec |
| `visualization.plot_gbdraw` | `executable` | Host executable/location or explicit Python publication destination |
| `visualization.plot_gbdraw` | `runner` | Python injection or rich filter object; passed unchanged |
| `visualization.plot_gbdraw` | `session` | Host executable/location or explicit Python publication destination |
| `visualization.plot_gbdraw` | `_op` | Private Python routing argument; not an Agent scientific option |
| `visualization.plot_gbdraw` | `_extra_flags` | Private Python routing argument; not an Agent scientific option |
| `visualization.plot_gene_structure` | `kwargs` | Native variadic escape hatch; no closed JSON field |
| `visualization.plot_gfa_graph` | `executable` | Host executable/location or explicit Python publication destination |
| `visualization.plot_gfa_graph` | `runner` | Python injection or rich filter object; passed unchanged |
| `visualization.plot_ogdraw_map` | `kwargs` | Native variadic escape hatch; no closed JSON field |
| `visualization.plot_pan_circular` | `bandage_executable` | Host executable/location or explicit Python publication destination |
| `visualization.plot_pan_circular` | `bandage_runner` | Python injection or rich filter object; passed unchanged |
| `visualization.plot_presence_tracks` | `coverage` | Native structured object; requires a designed closed JSON/artifact codec |
| `visualization.plot_sv_frequency` | `coverage` | Native structured object; requires a designed closed JSON/artifact codec |
| `visualization.plot_sv_schematic` | `schematic` | Native structured object; requires a designed closed JSON/artifact codec |
| `visualization.plot_track_density` | `regions` | Native structured object; requires a designed closed JSON/artifact codec |
| `visualization.plot_transfer_tracks` | `plastid_candidates` | Native structured object; requires a designed closed JSON/artifact codec |

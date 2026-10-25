# 已注册操作与公开 Python 函数的对应

2026-10-16。phytoorga-web 接入时，工具目录从注册表读参数 schema，后台任务调用公开 Python API（`ov.read` → `ov.<模块>.<函数>` → `ov.write`）。这要求每个被网站使用的操作都能找到一个**接受其 schema 全部参数**的公开函数。本文记录逐项核对的结论；规则由 `tests/operations/test_public_functions.py` 固定。

## 规则

- 操作 ID `<suite>.<name>` 默认对应 `ov.<suite>.<name>`。
- 少数操作注册的是给 Agent 用的 JSON 扁平包装，而 Python 侧已有人在用的函数名字不同。它们列在 `organelleverse.operations.PUBLIC_FUNCTIONS`，查询入口是 `ov.operations.public_function(operation_id)`（返回 `"<suite>.<function>"`，没有则为 `None`）与 `resolve_public_function`（返回函数对象）。
- 不为这些操作加同名别名：OV 坚持一个函数一个名字（`ov.save` 被禁止，`ov.fetch.accessions`/`entrez`/`refseq` 已移除），第二种拼写会让同一功能有两个入口。
- 只要一个操作能解析到公开函数，该函数就必须接受 schema 的全部参数（或 `**kwargs`）。`fetch.entrez_query` 曾违反这一点，见下文。

## 网站报告的 16 个操作

| 操作 | 结论 | 公开函数 |
| --- | --- | --- |
| `fetch.accession_list` | 注册的是 `_agent_ops.op_accession_list` 包装；Python 侧同参数函数已存在 | `ov.fetch.fetch_accessions`（表内映射） |
| `fetch.ngdc_gwh` | 同上 | `ov.fetch.fetch_ngdc`（表内映射） |
| `fetch.gene_records` | 同上 | `ov.fetch.fetch_genes`（表内映射） |
| `rna_editing.quantify_efficiency` | 注册函数本身就是公开实现，只是名字不同且未从套件导出 | `ov.rna_editing.quantify_known_site_efficiency`（新导出 + 表内映射） |
| `rna_seq.quantify_exon_expression` | `rna_seq` 套件已注册操作但未在根公开 | `ov.rna_seq.quantify_exon_expression`（`rna_seq` 设为公开模块） |
| `rna_seq.quantify_splicing_efficiency` | 同上 | `ov.rna_seq.quantify_splicing_efficiency` |
| `heteroplasmy.quantify_variant_fractions` | `heteroplasmy` 已注册但未在根公开 | `ov.heteroplasmy.quantify_variant_fractions`（设为公开模块） |
| `heteroplasmy.estimate_structural_fractions` | 实现位于 `heteroplasmy_structural` | `ov.heteroplasmy.estimate_structural_fractions`（由 `heteroplasmy` 重新导出） |
| `phylogeny.run_astral` | 同名函数存在于 `phylogeny.astral`，未从套件导出 | `ov.phylogeny.run_astral`（新导出） |
| `localization.score_nls` / `score_pts` / `score_signal_peptide` / `score_transit_peptides` / `score_transmembrane` | 同名函数存在于 `localize_core`，未从套件导出 | `ov.localization.score_*`（新导出） |
| `ppr.predict_binding_sites` | `ppr` 包已导出该函数，但未在根公开 | `ov.ppr.predict_binding_sites`（`ppr` 设为公开模块） |
| `morphology.batch_measure` | 注册的是 JSON 扁平入口 `batch_measure_dir`；同模块的 `batch.batch_measure` 接收 `BatchItem` 列表，签名不同，不能作为该操作的公开函数 | `ov.morphology.batch_measure_dir`（新导出 + 表内映射） |

结论：16 个都应该有公开函数，没有一个是“有意只给注册表用”。另外同类的 `fetch.nuclear_assembly` 也加入映射（`ov.fetch.fetch_nuclear_genome`）。

## `fetch.entrez_query` 参数对齐

注册操作接受 `min_length`、`max_length`、`refseq_only`、`min_gene_count`、`max_ambiguous`，公开 `ov.fetch.entrez_query` 原来只有 `min_length` 和 `filters=FetchFilters(...)`。现在公开函数直接接受这五个关键字（是 `FetchFilters` 同名字段的快捷方式），注册包装原样传入。关键字与 `filters` 中已设置的值冲突时报 `fetch.filter_conflict`，不再静默以某一方为准。

## 仍只能经注册表调用的 42 个操作

以下按名称和所在模块归类，未逐项审阅科学用途；如有外部调用者需要，按本文规则逐个补公开函数并扩展测试。

- 模型内部步骤（私有模块 `_plantc2u`、`_deepred`、`_bxb_extract`）：`rna_editing.change_n`、`encode_matrix`、`extract_window`、`extract_windows`、`plantc2u_model_path`；`coevolution.bxb_lengths_dlcpar`、`bxb_lengths_lca`、`filter_outlier_tree`、`normalize_edge_mean`、`normalize_gene_sum`、`species_tree_edges`。
- 后端安装与探测：`localization.check_all_backends`、`check_backend`、`install_hint`；`morphology.check_all_backends`、`check_backend`、`default_executor`、`install_hint`；`selection.install_hint`。
- 已弃用的读入：`io.read_fasta_genome`、`io.read_genbank_genome`、`io.read_long_reads`（用 `ov.read` / `ov.<模块>.read`）。
- 基准与参数搜索：`pangenome.evaluate_block_benchmark`、`evaluate_conformation_benchmark`、`evaluate_graph_benchmark`、`evaluate_taxonomy_benchmark`；`optimization.grid`、`random`、`space_filling`、`successive_halving`。
- 资源与输入文件写出：`phenotype.cms_protein_fasta`、`cms_reference_json`、`download_cms_sequences`；`selection.write_branch_model`、`write_branch_site_model`、`write_clade_model`、`write_codeml_inputs`、`write_site_model`。
- 其他：`codon_composition.compute_enc`、`resolve_genetic_code`、`phylogeny.pairwise_distances`、`tools.adapter`（插件工具适配器）。

## 同日核对的其他接口问题

- `rna_editing.predict_edits` 的 `backend` 现在是 `Literal["auto", "deepred", "plantc2u", "heuristic"]`，schema 带 `enum`。`prep` 不是该参数的取值，而是独立操作 `rna_editing.predict_edits_prep`。
- `plot_genome_map` / `plot_ogdraw_map` 的结果 `scope` 原来写死为 `mitochondrion`；现在取自 GenBank source 的 `/organelle`：全部为质体则 `plastid`，全部为线粒体则 `mitochondrion`，不一致为 `mixed`，均未声明为 `none`。
- `ov.annotation.annotate` 只接受带序列制品的基因组（由 FASTA 读入），这是有意设计：GenBank 读入得到的是已有注释，供 `annotation.extract`、`qc.annotation` 使用。报错现在会指出用 `ov.fetch.genbank_to_fasta` 写出 FASTA 后再读入。
- OVASM 不登记在 `ov.environments` 注册表；查找顺序见 README 的 `ov.environments` 小节。

# 分化时间估计：`phylogeny.date_tree`

Python 公共 API 和 capability 使用同一入口，输出 `OrganelleResult`。输入为
Newick 树文件、FASTA 比对和 `Calibration` 列表。已有 IQ-TREE `.treefile`
可直接与原比对一起使用；`partition_nexus` 可传入 `.best_scheme.nex`。
默认后端为 `backend="iqtree_lsd2"`：固定输入拓扑并重新估计分支长度，
随后运行 IQ-TREE 内置 LSD2。以下说明针对该默认后端；MCMCTree 见文末链接。

```python
from organelleverse.phylogeny import Calibration, date_tree

result = date_tree(
    "inferred.treefile", "alignment.fasta",
    calibrations=[
        Calibration(root=True, min_age_ma=90, max_age_ma=120,
                    source="填写文献及节点依据"),
        Calibration(mrca=["TaxonA", "TaxonB"], min_age_ma=70,
                    source="填写化石出处"),
    ],
    outgroup=["OutgroupA", "OutgroupB"],
    ci_replicates=100,
)
print(result.metrics["timetree_newick"])
print(result.metrics["node_ages"])
```

Agent JSON 中每项校准是同名字段的对象。`root=True` 与两个不同叶名的
`mrca` 二选一；至少有一个年龄边界，至少一个校准具有正的最小年龄以确定
绝对时间尺度。年龄单位为 Ma（距今百万年）。边界相同表示固定年龄；省略
一个边界表示单侧约束。拒绝未知叶名、重复节点、倒置范围和祖先/后代冲突。
`source` 原样保留，调用者负责校准文献及系统发育节点归属。

`distribution` 可省略或写 `"bounds"`。LSD2 只有硬边界，不解释概率密度，
因此 `normal`、`lognormal`、`uniform` 等先验明确报 `OrganelleInputError`。
不能把某文献的贝叶斯先验直接冒充 LSD2 先验。若采用该文献的范围，必须
说明这里使用的是硬边界。

树根为二分时信任其已有定根；其他情况必须给出外群。外群必须形成树的一侧，
不能跨越无关分支。IQ-TREE 使用可逆替换模型时会先去根，通过 `-o` 和 LSD2
的 `-r k` 保留指定根分支，允许在该分支上优化根的位置。所有末端年龄固定为
0，因此当前版本不支持古 DNA 或带采样日期的异时末端。

IQ-TREE 2.4 的原始 `.timetree.nwk` 实际保存替换量，必须除以报告速率才能得到
时间；`.timetree.nex` 的分支才直接是 Ma。本接口优先读 NEXUS 的 date 和
CI_date，另写 `dated_tree.nwk`，不会把原始替换量树当作时间树。

输出包括时间树 Newick（分支单位 Ma）、内部节点年龄表、每个节点的后代叶名、
校准、IQ-TREE 版本、完整 argv、LSD2 原始报告及树文件。节点 ID 为本次时间树
的前序序号；跨运行比较应匹配后代叶集合。`ci_replicates=0` 时区间为 `null`。

置信区间来自 LSD2 分支长度重采样（Poisson 加 lognormal 速率扰动），不是
序列 bootstrap，也不是贝叶斯 HPD。`clock_sd=0.2` 为软件默认值，仅控制区间
模拟的速率扰动，不能把它理解为点估计采用了完整的松弛钟模型。区间条件于
拓扑与校准，不能反映所有系统误差。[IQ-TREE 官方定年文档](https://iqtree.github.io/doc/Dating)

依赖沿用核心 Biopython 的树解析/重定根和现有 Pydantic 的结构模型，无新增
Python/Rust 依赖。IQ-TREE/LSD2 是完整推断程序，统一经 `run_external` 调用。
工具路径由 `ORGANELLEVERSE_IQTREE_BIN` 或 PATH 指定；底层
`organelleverse.phylogeny.dating.date_tree` 还支持 `iqtree_bin` 和新的
`output_dir`，公共 API 自动分配托管目录。`dry_run=True` 只返回 warning 计划，
不创建目录、不估计年龄。

本机 IQ-TREE 3.0.1 已复现正常退出但不运行 LSD2 的问题，显式 `--dating LSD`
也没有生成时间树；上游 [issue #63](https://github.com/iqtree/iqtree3/issues/63)
有同类报告。接口检查真实输出，缺失即报执行错误，不切换后端或返回空成功。
可以显式安装并指定独立环境中的 IQ-TREE 2.4.0：

```bash
micromamba create -n ov-dating -c conda-forge -c bioconda iqtree=2.4.0
export ORGANELLEVERSE_IQTREE_BIN=/your/envs/ov-dating/bin/iqtree2
```

MCMCTree 松弛钟现可通过 `backend="mcmctree"` 选择；软校准、双链收敛检查、
先验单位与支持范围见 [MCMCTree 使用说明](mcmctree-dating.md)。

## LSD2 退化解与校准边界

真实的受约束最小二乘解可能包含零时间枝。结果现在通过既有 `status="warning"`
和 flags 报告 `lsd2_zero_time_branches`、`lsd2_calibration_boundary`，以及
`lsd2_uncalibrated_node_collapsed`。最后一项意味着未校准内部节点沿零/近零枝
与已校准节点相连且年龄在阈值内相同；该年龄不能视为单独解析出的节点年代。
检测会遍历整段相连的零长度枝，不仅检查直接父子。边界命中本身是诊断提示，
不等同于所有节点都没有信息；固定年龄校准必然会命中边界。

`metrics.degeneracy` 提供 `zero_time_branches`（父/子 ID、是否末端、时间长度）、
`calibration_boundary_hits`（节点、上下界及年龄）、`affected_nodes`（内部节点 ID、
后代叶集合、是否校准、与哪些校准节点同龄）及 `collapsed_uncalibrated_node_ids`。
末端枝若接近零，也记录在枝列表中，末端 ID 使用 `tip:<叶名>`。

阈值取输出中最大内部年龄的 **六位有效数字最后一位的半个单位**：
`0.5 * 10**(floor(log10(max_age_ma)) - 5)` Ma，零年龄树取 0。
这是 IQ-TREE 2.4.0 / LSD2 日期打印的舍入分辨率，近零指在该绝对年龄精度下
无法区分，不代表生物学上严格同时发生。以 123.452 Ma 为例，阈值为 0.0005 Ma，
也能识别原始上界 123.4522 Ma 被打印为 123.452 的情况。此阈值事先由格式决定，
不依据期望年龄调整；不裁剪、合并或平滑任何枝长/年龄。

独立调用 `parse_timetree(..., calibrations=[...])` 时才有校准边界与同龄校准诊断；
不传校准仍报告零/近零枝。常规 `date_tree` 自动传入校准。真实两组回归结果和
MCMCTree 多分区比较见 [验证记录](dating-partitions-validation.md)。

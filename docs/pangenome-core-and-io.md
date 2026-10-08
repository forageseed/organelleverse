# 图 PAV 阈值、PGGB 参数与大读长 I/O

## core 的计数单位

`pangenome.analyze_graph(..., core_threshold=0.95)`、`WorkflowRequest`、
`graph.node_pav` 和 `pav_store.write_node_pav` 使用同一个节点分类规则：

- core：`core_threshold <= f <= 1`；
- shell：`cloud_threshold < f < core_threshold`；
- cloud：`0 < f <= cloud_threshold`；
- unobserved：`f == 0`。

要求 `0 <= cloud_threshold < core_threshold <= 1`。保留现有严格 core 默认值
1.0，避免改变旧分析的分类结果；软 core 必须显式传入，例如 0.95。
sample 表的分母为全部生物学样本，同一样本的多个路径按 OR 合并；path 表的
分母为全部路径。阈值进入 PAV metadata、summary、工作流参数、分析说明和结果
provenance。分类不改变二进制 PAV，也不筛选构树节点。

`classify_sequences` 是另一种测量：以 k-mer 重叠阈值判断整条序列在**其他**
基因组中是否存在，分母排除当前基因组。它保留默认 0.9，并输出该分母和存在
判据。它的 core/variable/private 不能直接等同于图节点 core/shell/cloud。

## PGGB 叶绿体片段下限及参数采用

叶绿体 PGGB 使用 `max(请求或推导的 segment_length, 5000)`。
依据为 2026-09-24 的 192 个禾本目叶绿体 benchmark：p84+s5000 的六个受检科
全部单系，RepeatMasker 推导的 s500 对照中莎草科和禾本科不是单系。
这是该 benchmark 支持的经验保护值，不表示对所有谱系的全局最优保证。
线粒体及其他构图后端不应用该下限。直接 Python builder 与受管理 builder
均执行下限，metrics 记录请求值及实际值，provenance 的参数标识覆盖这些值；
受管理结果另有 `parameter_policy`；采用推荐时，其 `derived_segment_length`
从所附重复证据恢复保护前的估计，`segment_length` 记录实际构图值。

推荐仍来自现有 Mash 距离和可选 RepeatMasker 重复区间估计。原始推导值与
保护后推荐值保存在推荐结果的 `metrics.segment_length_policy` 中；推荐 payload
的策略版本为 `sample-mash-plastid-floor.v3`，`segment_length` 为保护后的值。
策略版本变化会使旧推荐缓存不再命中新请求。

自动采用默认关闭。公开 API 可先取得推荐，再交给构图：

```python
recommendation = ov.pangenome.recommend_parameters(genomes, run_repeatmasker=False)
result = ov.pangenome.build_graph(
    genomes,
    method="pggb",
    recommendation=recommendation.model_dump(mode="json")["metrics"]["recommendation"],
    auto_adopt_recommendation=True,
)
```

此开关要求 PGGB 和显式提供的推荐，校验来源、输入内容标识及推荐完整性后才
采用其中的 identity 和 segment_length。没有开关时仍要求显式参数与所附推荐
完全一致。低于叶绿体下限的旧推荐报错并要求重新生成，避免把修改过的参数
宣称为原推荐的精确采用。图拥有独立 adoption artifact。

工作流中同时设置 `recommend_parameters=True`、`auto_adopt_recommendation=True`
即可生成后采用；RepeatMasker 仍由 `run_repeatmasker` 单独控制。图构建成功后
才写入采用证据，失败不会产生采用成功声明。

## 读长完整性与 I/O

读长统计在缓存未命中时，将原有“首次哈希 → 解析 → 末次哈希”三次完整读取
变为“边读边哈希并解析 → 末次哈希”两次。gzip 的哈希仍描述磁盘压缩字节，
统计描述解压后记录，包括多 gzip member。首次哈希与解析共用字节流，结果
不可能绑定到另一次读取的内容。

序列内空白校验从 Python 逐字符 `str.isspace()` 循环改为标准库正则 `\s`
扫描；测试遍历全部 Unicode 码点确认判据完全等价，FASTA/FASTQ 的接受规则
不变。此项减少统计阶段 CPU 开销，与减少读取次数分别报告。

保留末次独立哈希，以检测解析期间源路径被替换或内容改变。统计缓存命中也
仍先重算完整内容哈希；缓存键继续包含内容标识、解析器版本和统计 schema。
没有添加基于路径/大小/mtime 的可信哈希缓存：这些元数据相同不能证明内容
没有变化。输入 artifact 建立、组装复用边界及后端入口的校验也保留。

PMAT 2.1.5 的 `fq2fa` 使用 `kseq/gzopen`，只读支持普通及 gzip FASTA/FASTQ。
单文库且跳过纠错时直接传入绝对源路径，避免解压和写出整份输入；后端执行前
和收集结果时均重新校验源内容，变动则拒绝结果。输入应在运行期间保持不变；
和既有前后校验一样，这不是对恶意进程修改后恢复文件的隔离机制。
多个文库需要合并，纠错路径需要独立暂存，二者保留原有物化流程。

没有新增依赖、内容标识方案或 capability；使用 Python 标准库的流式 I/O、gzip
和已有哈希约定。大型数据的验证数字及实际测试命令见本任务 DELIVERY.md。

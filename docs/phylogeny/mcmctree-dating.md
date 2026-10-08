# MCMCTree 松弛钟定年

`phylogeny.date_tree` 保留默认 `backend="iqtree_lsd2"`，新增
`backend="mcmctree"`。这仍是同一个 capability，不新增 ID 或覆盖分类。
公共入口自动选择托管输出目录；需要指定目录的 Python 工作流使用领域函数：

```python
from organelleverse.phylogeny import Calibration, MCMCTreeOptions
from organelleverse.phylogeny.dating import date_tree

result = date_tree(
    "topology.nwk", "alignment.fa",
    [Calibration(root=True, min_age_ma=90, max_age_ma=125,
                 source="请填写本数据适用的化石或二次校准出处")],
    output_dir="new-dating-run",
    backend="mcmctree",
    mcmctree_options=MCMCTreeOptions(
        clock=2, burnin=50000, sampfreq=50, nsample=20000,
        rgene_gamma=[2.0, 20.0], sigma2_gamma=[2.0, 10.0],
    ),
    seed=43,
)
```

上例年龄仅演示语法，不是通用校准。可指定单系外群 `outgroup`；没有外群时，
与 LSD2 一样信任输入的二叉根。MCMCTree 要求整棵树完全二叉、现生末端年龄为零，
并要求根节点有显式年龄上界。缺工具抛 `OrganelleDependencyError`，运行失败或
不完整输出抛 `OrganelleExecutionError`，不会返回空年龄表或切换后端。

## 安装与支持范围

使用已有 PAML 完整软件，不重写采样器或近似似然算法。外部调用统一经
`core.external.run_external`；PAML 的 `usedata=3` 在内部调用 BASEML。
设 `ORGANELLEVERSE_MCMCTREE_BIN=/absolute/path/to/mcmctree`，或将工具加入 PATH；
领域函数也允许 `mcmctree_bin`。BASEML 从所选 MCMCTree 所在目录及 PATH 查找。
可独立安装：`micromamba create -n ov-mcmctree -c conda-forge -c bioconda paml`。
已在 PAML 4.10.9 验证。

支持单分区或通过 `partition_nexus` 指定多分区，各分区均为核苷酸 `GTR+G4`。
若 NEXUS 的 `charpartition` 指定其他模型，会明确拒绝。`threads`、`ci_replicates`、
`clock_sd` 是 LSD2 参数，不控制 MCMCTree。MCMCTree 每条链单线程，按序运行。
`clock=2` 是独立速率，`clock=3` 是相关速率。

## 分区输入

将以下内容保存为 `codon12-3.nex`，并传入 `partition_nexus="codon12-3.nex"`：

```nexus
#NEXUS
begin sets;
  charset codon12 = 1-25674\3 2-25674\3;
  charset codon3 = 3-25674\3;
end;
```

上例用于 25,674 bp、从密码子第一位开始的编码比对。三分区方案改为
`charset pos1 = 1-25674\3;`、`pos2 = 2-25674\3;`、`pos3 = 3-25674\3;`。
按基因分区则每个基因一个连续范围，例如 `charset atpB = 1-1500;`，后续
范围应取自该比对的基因边界表。包的 `build_partitioned_supermatrix` 生成的
`partitions.nex` 也可直接使用。调用者负责确保密码子阅读框正确；程序不会
从普通 DNA 比对猜测基因边界或阅读框。跨基因使用全局位置要求各基因从第一位
开始且长度均为三的倍数；否则应按各基因起点明确组合 charset。

所有 charset 必须非空、名称唯一、范围合法，并将比对每列恰好覆盖一次。
按声明顺序写成连续的 PAML 序列块，每块都有自己的样本数、长度和所有样本；
块内按原比对列顺序排列，保留缺口及模糊字符（`cleandata=0`）。`ndata` 等于
块数；每块独立拟合 GTR 频率、交换率和 Gamma 形状，分别计算枝长、梯度、
Hessian，共享拓扑、节点年龄和校准。结果的 `n_partitions`、`partitions`
记录每块名称、原 charset、位点数、模型；未提供分区文件时仍为单块。

## 校准、单位和先验

外部 API 的年龄均为 Ma；PAML 内部一个时间单位固定为 **100 Ma**。
输入 `Calibration` 在定根后按 MRCA 解析：

| 输入 | PAML 标记（年龄除以 100） |
|---|---|
| 下界与上界 | `B(min,max)` |
| 仅下界 | `L(min)` |
| 仅上界 | `U(max)` |

使用 PAML 默认软尾：`B` 两端各 0.025、`L` 下尾 0.025、`U` 上尾 0.025；
`L` 的另两个默认参数为 offset=0.1、scale=1。上下界相等仅适用于 LSD2，
MCMCTree 明确拒绝。输入 `distribution=None`/`"bounds"` 都使用上述后端语义；
没有把 LSD2 硬边界误称为贝叶斯先验。树上祖先顺序会改变有效边缘先验，
因此边界并不意味着节点后验必须落在其中。

Gamma 参数按 **shape、rate** 指定，均须正且有限。默认整体速率先验均值
2/20=0.1 substitutions/site/100 Ma，速率方差先验均值 2/10=0.2。
控制文件明确记录 `rgene_gamma = 2 20 1 0`、`sigma2_gamma = 2 10 1`，
以及固定 `BDparas = 1 1 0.1 c`（条件构造）。这些默认值不是数据估计值。
调用者应依据研究设计选择先验；正式科学推断还应检查无数据有效先验和先验敏感性。

多分区使用 PAML 的 **gamma–Dirichlet** 先验（第四项 `prior=0`），对分区
平均速率使用上述 Gamma，对分区间比例使用对称 Dirichlet（第三项浓度为 1）；
速率方差同样使用 gamma–Dirichlet。它不是为每个分区独立复制 Gamma 先验。
本次保持已有 shape/rate 和浓度不变，不根据目标年代调参；改变分区数仍会改变
联合速率模型，不能视为完全相同的先验。密码子分区也不意味着独立遗传的基因座。

## 两步法与结果

1. `likelihood/`：`usedata=3`，PAML 为每个分区调用 BASEML 估计替换枝长、梯度、
   Hessian，将各分区信息依次收集到 `out.BV`。
2. `chain1/`、`chain2/`：分别复制为 `in.BV`，以 `usedata=2 in.BV` 运行，
   独立随机种子为 `seed`、`seed+2`，使用 PAML 默认 arcsin 近似变换。

每个目录保留控制文件、比对、带校准树、stdout/stderr、PAML 原始输出。
输入叶名通过顺序短名传给 PAML，`metrics.taxon_aliases` 记录对应表；结果恢复原名。
输出目录必须是新目录，防止混入旧输出。`dry_run=True` 只返回 warning 计划，
不创建目录，不声称估计了年龄。

`node_ages` 包含 `age_ma`（两链等权后验均值）、`hpd_lower_ma`、
`hpd_upper_ma`、`descendant_taxa`、`is_root`。为兼容现有年龄表也提供
`ci_lower_ma`/`ci_upper_ma`，但 `ci_method` 明确说明这是后验 HPD。
95% HPD 用排序样本中覆盖至少 95% 样本的最短连续区间估计；多峰后验可能不适合
用单区间概括。`dated_tree.nwk` 枝长是后验均值年龄差，单位 Ma。

PAML 将预热样本排除在 `mcmc.txt` 外；当 `sampfreq>1` 时它还多写一个
`Gen=1` 记录。解析器验证完整采样序列，删除这一条额外记录，使 ESS 输入的时间
间隔均匀。每条链使用恰好 `nsample` 个样本；原始文件不修改。

## 收敛不是默认成功

对全部节点年龄、整体速率、速率方差及 lnL 分别检查：

- 每条链 ESS ≥200；采用 Geyer 初始正值、单调成对自相关序列，FFT 计算自协方差。
- 将两链各一分为二，常规 split R-hat ≤1.01。
- 输出每条链均值，以及两链均值绝对差；节点年龄差的单位是 Ma。

恒定序列 ESS=0，R-hat 无法定义时记 null 并视为未通过。任一参数未通过，
结果为 `status="warning"`、`flags=["mcmc_not_converged"]`，年龄表仅供诊断。
不会通过裁剪年龄、放宽阈值或自动调先验将其变成成功结果。
这些是数值收敛检查，不是模型正确性证明；未提供 rank-normalized R-hat 或 tail ESS。

算法与参数依据：
[PAML 手册](https://github.com/abacus-gene/paml/blob/master/doc/pamlDOC.pdf)、
[dos Reis、Álvarez-Carretero 与 Yang 的近似似然教程，第 2 节](https://github.com/abacus-gene/paml/blob/master/doc/MCMCtree.Tutorials.pdf)、
[PAML 官方说明](https://github.com/abacus-gene/paml/wiki/MCMCtree)、
[PAML 实现](https://github.com/abacus-gene/paml/blob/master/src/mcmctree.c)。
真实禾本目对比见同目录 `mcmctree-dating-validation.md`。
多分区与 LSD2 退化检测的后续验证见 `dating-partitions-validation.md`。

# 细胞器结构图谱

`ov.visualization.plot_structure_map` 在现有 OGDraw 圆图上增加 CPGView/PMGmap 风格的剪接面板和重复轨道，支持叶绿体与线粒体。无需安装新依赖。

```python
import organelleverse as ov
from organelleverse.structure import compute_repeats, compute_multiconf

ssrs = compute_repeats("genome.fa")["ssrs"]
repeats = compute_multiconf("genome.fa", min_repeat_len=50)["repeats"]
plot = ov.visualization.plot_structure_map(
    "genome.gb",
    genome_name="Arabidopsis thaliana",
    show_ir=True,  # 仅叶绿体；线粒体省略
    ssrs=ssrs,
    dispersed_repeats=repeats,
    # tandem_repeats=[{"start": 100, "end": 160}],  # 已计算的串联重复区间
)
result = ov.write(plot, "structure.svg")
ov.write(plot, "structure.png")
```

创建 `plot` 不写文件；写出时生成图片及同目录的 `structure.exons.tsv`，两者均列入返回结果的 artifacts。`plot.metrics["features"]` 保存绘图使用的全部注释块，可在写文件前检查。SVG 中每个剪接面板外显子有 `exon-<feature_index>-<exon>` 标识，反式连接有 `splice-<feature_index>-<left>-<right>` 标识。`feature_index` 是 GenBank 全部 features 中的零基序号，能够区分同名拷贝。

新增 capability：`visualization.plot_structure_map`。绑定采用 canonical Result，以遵守计算阶段不写文件的边界；它返回可序列化的绘图数据。现有 visualization writer 可直接根据完整的结构图数据恢复渲染，因此 capability/Registry 返回值以及 JSON 往返序列化后的结果同样支持 `ov.write`，不需要重新读取 GenBank。

## 坐标与科学语义

- CDS、tRNA、rRNA 的 location parts 按转录本顺序绘制，坐标为 **1-based inclusive**。保留重复拷贝和每一段自己的正负链，优先于覆盖整个内含子区间的 gene 注释；没有产品注释的 gene 仍显示在圆图，并在表中以 `feature_type=gene` 区分。
- 反式剪接依据产品或所属 gene 的 `/trans_splicing` 标记。所属 gene 通过 locus_tag/名称和带链的区间包含关系匹配；不会按距离或基因名称猜测。未标记的反式剪接不会被自动推断。
- 顺式面板保留真实外显子和内含子长度，每行各自按比例缩放，并标注内含子 bp 数。反式面板保留外显子长度比例，省略基因组间距（`//`）；虚线表示该转录本的相邻外显子连接，**不代表其中每个内含子都采用反式机制**。
- 同一外显子被不同转录本共用时，各转录本面板分别显示；核验计数因此是转录本中的外显子数，不是唯一基因组区间数。
- SSR/串联轨道接受现有分析的 `start/end` 区间；散在重复接受 `positions/length` 成对记录，绘制两份拷贝。坐标同样为 1-based inclusive，不在绘图阶段运行新的重复算法。
- `show_ir=True` 复用 main 的环形 IR 检测和 LSC/IRb/SSC/IRa 定向逻辑；metrics 中分区使用 **0-based start + length**，可跨原点。无法识别时明确报错。检测边界不保证与已发表边界逐碱基相同。

## 可复现验证

```bash
PYTHONPATH=$PWD/src /home/user/work/organelleverse/.venv/bin/python \
  scripts/validation/validate_structure_maps.py \
  /home/user/work/benchmarks/p1/feat-p1-structure-maps
```

使用仓库保存的 NCBI [NC_000932.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_000932.1) 与 [NC_037304.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_037304.1) GenBank 记录。参考核验走现有的独立文本/location 解析器，绘图走 Biopython；逐项比较基因、feature 序号、外显子序号、起止坐标、链，并核对 SVG 中实际存在的外显子图块。

叶绿体有 23 个剪接转录本、50 个外显子（21 个顺式、2 个 rps12 反式转录本）；线粒体有 9 个、32 个外显子（6 个顺式，nad1/nad2/nad5 各一个反式）。坐标、数量和 SVG 图块均无不一致。逐个基因的坐标与数量见验证目录中的 `*.independent_exon_check.tsv` 和 `validation.json`。

NC_000932.1 没有显式分区注释，因此 IR 使用[文献表 1](https://doi.org/10.3389/fgene.2023.1131644)的独立数值作比较，未将其冒称为 GenBank 分区注释：

| 分区 | 文献区间（1-based） | 检测区间（1-based） | 起点差 / 终点差 |
|---|---|---|---|
| LSC | 1–84170 | 1–84170 | 0 / 0 bp |
| IRb | 84171–110434 | 84171–110439 | 0 / +5 bp |
| SSC | 110435–128214 | 110440–128209 | +5 / −5 bp |
| IRa | 128215–154478 | 128210–154478 | −5 / 0 bp |

验证图中的 SSR 为叶绿体 232 条、线粒体 132 条；≥50 bp 的重复对分别为 1 和 80。示例串联轨道选用其中相邻的直接重复对，分别为 0 和 1 个区间；这不是所有长度串联重复的完整普查。没有修改检测阈值来消除 IR 的差值。

作为额外交叉检查，现有 `compute_multiconf(min_repeat_len=50)` 在叶绿体中识别的唯一精确反向重复对为 84,171–110,434 和 128,215–154,478，每份 26,264 bp，与文献 IR 区间一致。结构图的分区轨道仍忠实使用 main 的 IR 检测边界，因此保留上述 5 bp 差异。

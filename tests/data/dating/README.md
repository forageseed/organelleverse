# IQ-TREE/LSD2 真实输出 fixture

来源：2026-09-26，本任务只读使用
`~/work/benchmarks/p1/feat-p1-mrbayes-partition/` 中 28 个禾本目叶绿体的
38 基因 / 25,674 bp 比对、IQ-TREE 树和最佳分区；IQ-TREE 2.4.0 内置
LSD2 2.4.4，seed=42，threads=2，CI=100，clock_sd=0.2。

完整生成过程见 `scripts/validation/validate_divergence_dating.py`。
校准为凤梨科茎 90.8908–123.4522 Ma、冠 15.3615–37.7002 Ma
（Vera-Paz et al. 2023，doi:10.3389/fpls.2023.1205511），香蒲属茎最小
70 Ma（Zhou et al. 2018，doi:10.1038/s41598-018-27279-3）。
以上全按硬边界使用。

两个文件直接复制外部程序输出，没有修改数值或标签。NEXUS 含 Ma 分支和
带符号 date / CI_date；Newick 含拟合替换量分支。报告中的速率为
0.000343478 substitutions/site/Ma。CI 是重采样结果，重跑不要求逐位相同。

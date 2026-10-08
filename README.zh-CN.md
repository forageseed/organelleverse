<div align="center">

<img src="docs/assets/logo.png" alt="OrganelleVerse" width="140">

# OrganelleVerse

**面向 AI 的植物细胞器基因组分析基座。**

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

## 它是什么

OrganelleVerse 是一个底层平台，让 AI 助手（或者你自己）分析植物的线粒体和叶绿体基因组：从测序数据，
到注释好、检查过、比较过的基因组。

每个分析都是一个名字清楚、输入输出明确的函数。它调用真实的程序，某一步跑不了会直接告诉你，
并且把做过的事情完整记录下来。这样把它交给 AI 才放心：AI 调用函数，你能看懂结果、信任结果、也能重复结果。

- **取数据**：从公共数据库下载测序数据和基因组。
- **组装**线粒体或叶绿体基因组，并**注释**（基因、tRNA、rRNA）。
- **检查**：生成质量报告。
- **看清一个基因组**：重复序列、密码子偏好、RNA 编辑、反式剪接、IR 边界。
- **比较**多个基因组：基因组成、基因顺序、泛基因组、条形码。
- **研究进化**：进化树、分歧时间、选择压力、协同进化、群体、基因转移。
- **画图**：基因组图、进化树、热图；还能**测量**显微镜图像里的细胞器。

结果都是常见文件：FASTA、GenBank、GFF3、表格、图和可读的报告。

## 23 个分析模块

每个模块对应 `ov` 下的一个领域（比如 `ov.annotation`）。**自研**表示方法是本项目自己写、自己测的；
**整合现有程序**表示由本项目负责管理别人写的程序（运行目录、版本、清楚的报错），并在外面加上自己的检查。

| 模块 | 做什么 | 来源 |
|---|---|---|
| `assembly` | 组装线粒体和叶绿体基因组。可以调用 Oatk、HiMT、GetOrganelle、PMAT2，也有我们自己写的 **OVASM**：单个 Rust 可执行文件，原始数据只扫一遍，输出的是带每个连接点 reads 支持度的组装图（不只是一条序列），reads 分不出结构时会明说。OVASM 有自己的仓库 [forageseed/ovasm](https://github.com/forageseed/ovasm)：编译后放进 `PATH`，或用 `ORG_VERSE_OVASM_BIN` 指向它。目前仍在做跨物种测试。 | 自研（OVASM）+ 整合现有程序 |
| `annotation` | 找出线粒体和叶绿体基因组里的蛋白基因、tRNA 和 rRNA。缺什么、拒绝了哪些候选都会如实报告，不瞎猜。序列搜索用 LOSAT（或 BLAST+），参考数据随包提供。 | 自研 |
| `barcode` | 根据比对设计 DNA 条形码和引物，也能由序列鉴定物种。 | 自研（引物用 Primer3） |
| `codon_composition` | 密码子使用、RSCU、有效密码子数、GC3 和氨基酸组成。 | 自研 |
| `coevolution` | 进化速率协变（ERC）：找出一起进化的基因，比如细胞器和核基因里的搭档。 | 自研 |
| `comparative` | 基因组成、基因顺序（共线性）、沿基因组的一致性、结构变异、叶绿体方向统一。 | 自研 |
| `composition` | 沿基因组的 GC 含量和 GC skew。 | 自研 |
| `diversity` | 核苷酸多样性、Watterson θ、Tajima's D，也能做滑窗。 | 自研（用 scikit-allel 的估计量） |
| `hgt` | 供体和受体基因组之间的水平基因转移。 | 自研 |
| `ir_boundary` | 叶绿体反向重复区（LSC / IRb / SSC / IRa）的边界和每个边界上的基因，并出图。 | 自研 |
| `localization` | 预测蛋白在细胞里去哪里。 | 整合现有程序（DeepLoc、TargetP） |
| `morphology` | 在显微镜图像里找出并测量叶绿体、线粒体、液泡和细胞核；训练模型、修正标注。 | 自研（测量）+ 整合现有程序（分割） |
| `pangenome` | 构建图泛基因组、基因有无、核心和可变部分、子图、格式转换。 | 整合现有程序（minigraph、PGGB、PanTools）+ 自研分析 |
| `phenotype` | 筛选细胞质雄性不育（CMS）候选基因，并评估每个候选的证据。 | 自研 |
| `phylogeny` | 比对、修剪、最大似然和贝叶斯进化树、分歧时间、单倍型网络、祖先状态、树的比较。 | 整合现有程序（MAFFT、IQ-TREE、MrBayes、RAxML、LSD2、MCMCTree）+ 自研网络 |
| `population` | 群体分化（F<sub>ST</sub>）、变异检测、细胞器与核的关联、细胞器 DNA 在核里的拷贝。 | 自研 + 整合现有程序（DeepVariant、GEMMA） |
| `rna_editing` | 预测 C 到 U 的编辑位点，并在 RNA-seq 比对里检测和验证。 | 自研 + 现有模型（DeepRed-Mt、PlantC2U） |
| `selection` | Ka/Ks、密码子感知的比对，以及位点、分支、分支-位点、分支类群等选择压力检验。 | 自研（Ka/Ks）+ 整合现有程序（PAML、HyPhy、KaKs_Calculator） |
| `structure` | 简单重复序列、可能重组的重复对（多构型）、内含子，以及从组装图解析出的结构。 | 自研 |
| `trans_splicing` | 根据注释找出反式剪接基因。 | 自研 |
| `transfer` | 细胞器之间、细胞器到核的 DNA 转移，用比对、读深和长读长支持来判断。 | 自研 |
| `variation` | 比对中相对参考的 SNP 和 SNP 密度。 | 自研 |
| `visualization` | 基因组图、结构图、进化树、热图、共线性、ERC 和基因转移图、质量仪表盘。 | 自研（个别图也可以用现有的绘图程序） |

`ov` 里还有不算分析的通用功能：`fetch`（下载数据）、`qc`（质量检查）、`report`（HTML 报告）、
`format_conversion`、`read` 和 `write`。

## 安装

需要 Python 3.11 或更高版本。

```bash
git clone https://github.com/forageseed/organelleverse.git
cd organelleverse
pip install .
```

序列搜索（注释、基因转移等）只要找得到 [LOSAT](https://github.com/satoshikawato/LOSAT) 就优先用它
（BLAST 的 Rust 高速重实现），找不到时退回 NCBI BLAST+。要用 LOSAT，先编译（`cargo build --release`），
再把 `losat` 放进 `PATH`，或者用环境变量 `ORG_VERSE_LOSAT_BIN` 指向这个文件。

组装和部分分析还会调用其他现成的程序（Oatk、HiMT、GetOrganelle、PMAT2、IQ-TREE、PAML 等；[OVASM](https://github.com/forageseed/ovasm) 从它自己的仓库编译）。
OrganelleVerse 会用你电脑上已有的。想看它找到了哪些：

```bash
organelleverse-environments list
```

## 怎么查某个函数的用法

所有功能都挂在 `ov` 下面，按领域分组，剩下的让 Python 来告诉你：

```python
import organelleverse as ov

dir(ov)                              # 有哪些领域：assembly、annotation、qc、phylogeny……
dir(ov.phylogeny)                    # 某个领域里有哪些函数
help(ov.phylogeny.build_tree)        # 这个函数做什么、每个参数是什么意思
```

各个领域更详细的说明在 `docs/`。

## 快速上手

仓库在 `examples/data/` 里带了一个真实基因组：拟南芥线粒体基因组（NCBI RefSeq NC_037304.1），
有 FASTA 和带注释的 GenBank 两种格式。例子用的就是它；其他文件名是占位，换成你自己的数据。
后面的例子接着前面的往下写。每个调用返回一个结果对象，`ov.write(结果, "文件夹")` 把它存成文件。

### 基础：取数据、读、写、检查

```python
import organelleverse as ov

ov.fetch.entrez_query(organelle="mitochondrion", dest="data/", taxon="Oryza sativa")  # 在 NCBI 搜索
ov.fetch.fetch_accessions(accessions=["NC_037304"], dest="data/")                     # 或按登录号下载

fasta = "examples/data/arabidopsis_mitochondrion.fasta"
genome = ov.read(fasta, organelle="mitochondrion", species="Arabidopsis thaliana")    # 只有序列
annotated = ov.read("examples/data/arabidopsis_mitochondrion.gb",
                    organelle="mitochondrion", species="Arabidopsis thaliana")        # 带注释
```

`ov.read(...)` 也能读测序数据：`ov.read("sample.hifi.fastq.gz", technology="hifi")`。
`technology` 可以写 `"hifi"`、`"ont"`、`"clr"` 或 `"illumina"`（双端 Illumina 把两个文件一起传：
`["r1.fastq.gz", "r2.fastq.gz"]`）。不确定的话写 `technology="auto"`：它会根据读段名字判断测序平台，
判断不了就明说，不会瞎猜。

```python
ov.qc.read_statistics("sample.hifi.fastq.gz")     # 读段长度和质量
ov.format_conversion.convert(annotated)           # GenBank 转 GFF3 和 FASTA
```

### `ov.assembly`：用测序数据组装基因组

```python
data = ov.read("sample.hifi.fastq.gz", technology="hifi")
assembly = ov.assembly.assemble(data, organelle="mitochondrion")   # 组装软件自动选
ov.write(assembly, "results/assembly")

ov.assembly.check_all_backends()                  # 装了哪些组装软件
ov.qc.assembly(assembly)                          # 组装结果的质量检查
```

`organelle` 写 `"mitochondrion"`（线粒体）或 `"plastid"`（质体）。

OVASM 支持 HiFi、原始 ONT、原始 CLR 和 Illumina（单端或双端），以及**长读长 + Illumina 混合组装**
（ONT 或 CLR 加一份 Illumina：用准确的短读长校正带噪声的长读长）。混合组装时把两份数据一起传给
`ov.read(long_libraries=..., short_libraries=...)`。HiFi 是测得最充分的路线；纯 Illumina 和混合组装仍属实验性，
OVASM 本身不使用双端配对信息，只用短读长可能把基因组拆成多段，请查看组装摘要里的 `k_accepted` 和 `circular`。

### `ov.annotation`：基因、tRNA 和 rRNA

```python
annotation = ov.annotation.annotate(genome)       # 约一分钟：38 个蛋白基因、28 个 tRNA、3 个 rRNA
ov.write(annotation, "results/annotation")        # GenBank、GFF3、表格、cds.fasta、proteins.fasta

ov.annotation.find_orfs(fasta)                    # 开放阅读框
ov.annotation.extract(annotated)                  # 基因、CDS 和蛋白序列

report = ov.qc.annotation(annotation)             # 注释的质量检查
ov.write(report, "results/annotation_qc")         # Markdown 总结、CSV 表格、JSON
ov.report.build(annotated, [report])              # 一份 HTML 报告
```

打开输出文件夹里的 Markdown 文件就能看到通俗的总结，CSV 表格里是详细数据。

### `ov.barcode`：DNA 条形码

```python
ov.barcode.design("alignment.fasta")                       # 能区分物种的条形码区域
ov.barcode.identify("query.fasta", "references.fasta")     # 这条序列是哪个物种？
```

### `ov.codon_composition`：密码子使用

```python
ov.codon_composition.codon_usage("results/annotation/cds.fasta", organelle="mito")
ov.codon_composition.amino_acid("results/annotation/cds.fasta", organelle="mito")
```

### `ov.coevolution`：一起进化的基因

```python
ov.coevolution.run_erc(gene_trees, species_tree)  # 进化速率协变
```

### `ov.comparative`：比较基因组

```python
plastomes = {"Arabidopsis thaliana": "a.gb", "Citrus sinensis": "b.gb", "Amborella trichopoda": "c.gb"}
genomes = [ov.read(path, organelle="plastid", species=name) for name, path in plastomes.items()]

ov.comparative.compare_genes(genomes)             # 各基因组有哪些基因
ov.comparative.synteny(genomes)                   # 基因顺序的共线块
ov.comparative.detect_structural_variants("reference.fasta", "query.fasta")
```

### `ov.composition`：GC 含量

```python
ov.composition.gc_content(fasta)                  # 沿基因组的 GC 含量和 GC skew
```

### `ov.diversity`：遗传多样性

```python
ov.diversity.nucleotide_diversity("alignment.fasta")
ov.diversity.neutral_tests("alignment.fasta")     # Tajima's D 等
```

### `ov.hgt`：水平基因转移

```python
ov.hgt.detect("donor.fasta", "recipient.fasta")
```

### `ov.ir_boundary`：叶绿体反向重复区

```python
chloroplast = ov.read("chloroplast.gb", organelle="plastid", species="Arabidopsis thaliana")
ov.ir_boundary.ir_boundary([chloroplast])         # LSC / IRb / SSC / IRa 的边界
```

### `ov.localization`：蛋白在细胞里去哪里

```python
ov.localization.predict("results/annotation/proteins.fasta")
```

### `ov.morphology`：显微镜图像里的细胞器

```python
ov.morphology.segment("cells.tif")                # 找出叶绿体、线粒体、液泡、细胞核
ov.morphology.measure("labels.tif")               # 每个的大小和形状
```

### `ov.pangenome`：泛基因组

```python
ov.pangenome.gene_pav(genomes)                    # 核心基因和可变基因
fasta_genomes = [ov.read(path, organelle="plastid", species=name) for name, path in
                 {"Arabidopsis thaliana": "a.fasta", "Citrus sinensis": "b.fasta"}.items()]
ov.pangenome.build_graph(fasta_genomes)           # 构建泛基因组图（需要 FASTA 格式的基因组）
```

### `ov.phenotype`：雄性不育候选基因

```python
ov.phenotype.cms(fasta)                           # 细胞质雄性不育（CMS）候选基因
```

### `ov.phylogeny`：进化树和网络

```python
ov.phylogeny.align("genes.fasta")
ov.phylogeny.build_tree("alignment.fasta")        # 最大似然进化树
ov.phylogeny.haplotype_network("alignment.fasta")
```

### `ov.population`：群体

```python
ov.population.call_variants("bams/", ref_path="reference.fasta")
ov.population.compute_fst("variants.vcf", {"sample1": "north", "sample2": "south"})
```

### `ov.rna_editing`：C 到 U 的 RNA 编辑

```python
ov.rna_editing.predict_edits(annotated)           # 预测编辑位点
ov.rna_editing.detect_editing_sites("rnaseq.bam", "mito.fasta")   # 在 RNA-seq 读段里检测位点
```

### `ov.selection`：自然选择

```python
ov.selection.kaks("cds_pairs.fasta")              # Ka/Ks
ov.selection.site_model(alignment="codon.fasta", tree="tree.nwk")
```

### `ov.structure`：重复序列与结构

```python
ov.structure.repeats(fasta)                       # 简单重复序列
ov.structure.multiconf(fasta)                     # 可能发生重组的重复对
```

### `ov.trans_splicing`：反式剪接基因

```python
ov.trans_splicing.detect_trans_splicing(annotated)
```

### `ov.transfer`：基因组之间的 DNA 转移

```python
ov.transfer.mtpt("mito.fasta", "chloroplast.fasta")     # 线粒体里的质体来源 DNA
ov.transfer.detect(nuclear_fasta="nuclear.fasta", organelle_fasta="mito.fasta")
```

### `ov.variation`：SNP

```python
ov.variation.snp("alignment.fasta")
ov.variation.snp_density("alignment.fasta")
```

### `ov.visualization`：画图

```python
plot = ov.visualization.genome_map(["examples/data/arabidopsis_mitochondrion.gb"])
ov.write(plot, "results/genome_map.png")
ov.visualization.plot_tree("tree.nwk")
ov.visualization.heatmap({"gene1": {"gene2": 0.8}, "gene2": {"gene1": 0.8}})
```

## 几点说明

- **不藏问题。** 某一步跑不了（缺程序、数据太少），会直接告诉你原因，不会悄悄换个结果给你。
- **结果可追溯。** 每个输出文件夹都记录了用了哪些输入、程序和版本。
- **只有你写出时才落盘。** 计算在临时区进行，`ov.write(...)` 才会把最终结果放到你指定的位置。

## 反馈与参与

欢迎在 GitHub 上提问题、报 bug、提想法。

## 引用

目前还没有论文。请引用本仓库、版本号（1.0.0）以及你实际运行过的程序
（LOSAT、Oatk、HiMT、GetOrganelle、PMAT2、BLAST+、HMMER、NCBI 资源）。

## 许可证

MIT

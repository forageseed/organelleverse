"""Find the operation an Agent should call from a natural-language request.

The default registry holds about three hundred operations, many of them
pipeline stages or helpers beside the analysis a person asks for. Listing them
all, or matching substrings of their descriptions, sends an Agent to a parser
or an install hint as often as to the analysis. This module ranks operations by
whole-word matches against the identifier, title, keywords and description,
weights rare words above common ones, and uses each contract's ``visibility``:
``primary`` operations are offered first, ``step`` operations rank lower, and
``internal`` ones are never offered (their ``superseded_by`` names the
operation to call instead).

Chinese requests are matched through :data:`ZH_TERMS`, a small table of
domain terms, reviewed by the domain owner on 2026-10-16; extend it rather
than adding per-operation translations, and keep single broad words (构建,
读取, 形态) out: they appear in almost every request and skew the ranking.

The same ranking serves any caller: ``ov.operations.search`` for code-writing
Agents, and hosts that hold their own copy of the catalog through
:func:`rank`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .spec import OperationSpec, OperationVisibility

__all__ = ["OperationHit", "SearchEntry", "ZH_TERMS", "entry_for", "rank", "search"]

#: Chinese domain terms -> English words that the contracts use. Longest terms match first.
ZH_TERMS: dict[str, str] = {
    "GC含量": "gc content",
    "GC偏斜": "gc skew",
    "简单重复序列": "ssr repeats",
    "重复序列": "repeats",
    "散在重复": "dispersed repeats",
    "多构型": "multiconf configurations recombination",
    "重组构型": "multiconf configurations recombination",
    "注释": "annotate annotation",
    "组装": "assemble assembly",
    "质控": "qc quality",
    "质量控制": "qc quality",
    "质量评估": "qc quality",
    "系统发育树": "phylogenetic tree",
    "系统发育": "phylogenetic phylogeny",
    "进化树": "phylogenetic tree",
    "建树": "build tree",
    "构建系统发育树": "build phylogenetic tree",
    "构建进化树": "build phylogenetic tree",
    "构建泛基因组": "build pangenome graph",
    "系统树": "phylogenetic tree",
    "参考基因组": "reference genome",
    "多序列比对": "align alignment",
    "比对": "align alignment",
    "修剪比对": "trim alignment",
    "选择压力": "selection kaks",
    "正选择": "positive selection",
    "RNA编辑": "rna editing",
    "编辑位点": "rna editing sites",
    "编辑效率": "editing efficiency",
    "密码子偏好": "codon usage rscu",
    "密码子": "codon",
    "氨基酸组成": "amino acid composition",
    "核苷酸多样性": "nucleotide diversity",
    "遗传多样性": "nucleotide diversity",
    "中性检验": "neutral tests tajima",
    "线粒体": "mitochondrion",
    "叶绿体": "plastid",
    "质体": "plastid",
    "反向重复": "ir inverted repeat",
    "IR边界": "ir boundary",
    "共线性": "synteny collinearity",
    "泛基因组": "pangenome",
    "基因有无": "gene presence absence",
    "单倍型网络": "haplotype network",
    "单倍型": "haplotype",
    # NUMT / NUPT / MTPT under their usual Chinese names; the short forms stay as aliases.
    "核内线粒体DNA": "numt",
    "核内线粒体": "numt",
    "核线粒体": "numt",
    "核内质体DNA": "nupt",
    "核内质体": "nupt",
    "核质体": "nupt",
    "线粒体中的质体来源序列": "mtpt plastid derived",
    "质体来源序列": "mtpt plastid derived",
    "叶绿体来源序列": "mtpt plastid derived",
    "基因转移": "transfer",
    "水平基因转移": "hgt horizontal transfer",
    "异质性": "heteroplasmy",
    "基因组图谱": "genome map",
    "圈图": "circular map",
    "可视化": "plot",
    "画图": "plot",
    "绘图": "plot",
    "绘制": "plot",
    "亚细胞定位": "subcellular localization",
    "定位预测": "localization predict",
    "转运肽": "transit peptide",
    "信号肽": "signal peptide",
    "细胞器形态": "organelle morphology",
    "形态测量": "morphology measure",
    "分割": "segment segmentation",
    "细胞质雄性不育": "cms",
    "雄性不育": "cms",
    "下载": "fetch download",
    "读取基因组": "read genome",
    "测序数据": "reads",
    "短读长": "short reads",
    "长读长": "long reads",
    "读长": "reads",
    "过滤": "filter",
    "格式转换": "convert",
    "条形码": "barcode",
    "引物": "primers",
    "结构变异": "structural variants",
    "群体": "population",
    "内含子": "introns",
    "反式剪接": "trans splicing",
    "表达量": "expression",
    "剪接效率": "splicing efficiency",
    "效率": "efficiency",
    "变异位点": "variant sites",
    "分化时间": "date divergence",
    "分歧时间": "date divergence",
    "祖先状态": "ancestral states",
    "共进化": "coevolution",
    "开放阅读框": "orfs",
}

# Words that every contract or request carries and that therefore say nothing about which one.
_STOP = frozenset(
    "a an and the of for to in on from with by my our this that it its is are be as at or "
    "please use using want need get do does give show me how what which can run compute "
    "computing calculate calculating analyze analyse analysis result results restored "
    "find finding identify detect detection estimate "
    "file files data organelle organellar".split()
)
#: Spelling variants that contracts use under one form.
_CANON = {
    "mitochondrial": "mitochondrion", "mito": "mitochondrion", "mitogenome": "mitochondrion",
    "mitochondria": "mitochondrion", "mt": "mitochondrion",
    "chloroplast": "plastid", "plastome": "plastid", "cp": "plastid", "plastidial": "plastid",
    "phylogenetic": "phylogeny", "phylogenies": "phylogeny", "phylogenomic": "phylogeny",
    "ssrs": "ssr", "microsatellite": "ssr", "microsatellites": "ssr",
    "assemble": "assembly", "assembler": "assembly",
    "annotate": "annotation",
    "aligned": "alignment", "align": "alignment", "msa": "alignment",
    "trees": "tree", "newick": "tree",
    "draw": "plot", "figure": "plot", "visualize": "plot", "visualise": "plot", "render": "plot",
    "diagram": "plot", "chart": "plot",
    "dn/ds": "kaks", "dnds": "kaks", "omega": "kaks",
}
_TOKEN = re.compile(r"[a-z0-9]+(?:/[a-z0-9]+)?")
_VISIBILITY_WEIGHT: dict[str, float] = {"primary": 1.0, "step": 0.45}
_FIELD_WEIGHT = {"name": 3.0, "title": 2.0, "keywords": 2.0, "description": 1.0}
_ORGANELLES = frozenset({"mitochondrion", "plastid"})
_PLOT_FAMILY = "visualization."
_PLOT_WORDS = frozenset({"plot", "map", "visualization", "heatmap", "circo", "ideogram", "panel", "track"})


def _stem(word: str) -> str:
    word = _CANON.get(word, word)
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    return _CANON.get(word, word)


def _words(text: str) -> list[str]:
    text = text.lower().replace("_", " ").replace("-", " ").replace("π", " pi ").replace("θ", " theta ")
    words: list[str] = []
    for token in _TOKEN.findall(text):
        # "dn/ds" stays one word (canonical kaks); "numt/nupt" also yields each part.
        words.append(_stem(token))
        if "/" in token:
            words.extend(_stem(part) for part in token.split("/"))
    return words


def _query_terms(query: str) -> list[str]:
    text = query
    for term in sorted(ZH_TERMS, key=len, reverse=True):
        if term.lower() in text.lower():
            text = re.sub(re.escape(term), " " + ZH_TERMS[term] + " ", text, flags=re.IGNORECASE)
    seen: list[str] = []
    for word in _words(text):
        if word not in _STOP and word not in seen:
            seen.append(word)
    return seen


@dataclass(frozen=True)
class SearchEntry:
    """What the ranking reads of one operation; hosts build it from their own catalog rows."""

    operation_id: str
    title: str = ""
    description: str = ""
    keywords: tuple[str, ...] = ()
    visibility: OperationVisibility = "primary"
    organelle_types: tuple[str, ...] = ()
    superseded_by: str | None = None
    fields: dict[str, frozenset[str]] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        name = self.operation_id.replace(".", " ")
        object.__setattr__(self, "fields", {
            "name": frozenset(_words(name)) - _STOP,
            "title": frozenset(_words(self.title)) - _STOP,
            "keywords": frozenset(w for k in self.keywords for w in _words(k)) - _STOP,
            "description": frozenset(_words(self.description)) - _STOP,
        })


@dataclass(frozen=True)
class OperationHit:
    """One ranked operation: call it by ``operation_id``."""

    operation_id: str
    score: float
    visibility: OperationVisibility
    title: str
    summary: str
    matched: tuple[str, ...]


def entry_for(spec: OperationSpec) -> SearchEntry:
    return SearchEntry(
        operation_id=spec.operation_id,
        title=spec.title,
        description=spec.description,
        keywords=spec.keywords,
        visibility=spec.visibility,
        organelle_types=tuple(spec.organelle_types),
        superseded_by=spec.superseded_by,
    )


def _summary(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    first = re.split(r"(?<=[.;])\s", text, maxsplit=1)[0]
    return first if len(first) <= limit else first[: limit - 1] + "…"


def _matches(term: str, words: frozenset[str]) -> float:
    if term in words:
        return 1.0
    if len(term) >= 5 and any(word.startswith(term) or (len(word) >= 5 and term.startswith(word)) for word in words):
        return 0.6
    return 0.0


def rank(
    entries: Iterable[SearchEntry],
    query: str,
    *,
    limit: int = 10,
    organelle: str | None = None,
    include_steps: bool = True,
) -> list[OperationHit]:
    """Rank ``entries`` for ``query``; ``internal`` entries never appear.

    An organelle named in the query is context, not a topic: most contracts do
    not repeat it, so it filters out operations declared for the other
    organelle and adds a small bonus where it appears, without counting
    against how much of the query an operation covers.
    """
    terms = _query_terms(query)
    named = [term for term in terms if term in _ORGANELLES]
    # A request of organelle words only ("plastid-derived mitochondrial ...") ranks by them.
    terms = [term for term in terms if term not in _ORGANELLES] or named
    if organelle is None and len(named) == 1:
        organelle = named[0]
    pool = [
        entry for entry in entries
        if entry.visibility != "internal"
        and (include_steps or entry.visibility == "primary")
        and (organelle is None or not entry.organelle_types or organelle in entry.organelle_types)
    ]
    if not terms or not pool:
        return []
    document_frequency: Counter[str] = Counter()
    for entry in pool:
        document_frequency.update(set().union(*entry.fields.values()))
    total = len(pool)
    wants_plot = bool(_PLOT_WORDS & set(terms))

    def idf(term: str) -> float:
        return math.log(1.0 + total / (1.0 + document_frequency.get(term, 0)))

    hits: list[OperationHit] = []
    for entry in pool:
        score = 0.0
        matched: list[str] = []
        for term in terms:
            best = max(_FIELD_WEIGHT[name] * _matches(term, words) for name, words in entry.fields.items())
            if best:
                score += best * idf(term)
                matched.append(term)
        if not matched:
            continue
        coverage = len(matched) / len(terms)
        score *= coverage * coverage * _VISIBILITY_WEIGHT[entry.visibility]
        if any(term in words for term in named for words in entry.fields.values()):
            score *= 1.1
        if not wants_plot and entry.operation_id.startswith(_PLOT_FAMILY):
            # A figure of an analysis shares its words; offer it only when the request asks for one.
            score *= 0.5
        hits.append(OperationHit(
            operation_id=entry.operation_id,
            score=round(score, 4),
            visibility=entry.visibility,
            title=entry.title,
            summary=_summary(entry.description),
            matched=tuple(matched),
        ))
    hits.sort(key=lambda hit: (-hit.score, hit.visibility != "primary", len(hit.operation_id), hit.operation_id))
    return hits[: max(1, limit)]


def search(
    query: str,
    *,
    limit: int = 10,
    organelle: str | None = None,
    include_steps: bool = True,
    specs: Sequence[OperationSpec] | None = None,
) -> list[OperationHit]:
    """Rank default-registry operations (or ``specs``) for a natural-language ``query``."""
    if specs is None:
        from .registry import registry

        specs = registry.list()
    return rank((entry_for(spec) for spec in specs), query, limit=limit, organelle=organelle,
                include_steps=include_steps)

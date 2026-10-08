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

# molo 架构

本文档描述 molo 的目标架构：长期的分层边界、依赖方向与副作用执行模型。
它是给贡献者和集成者的指南，不是变更日志或设计争论记录。

## 定位

molo 是一个可嵌入的 Rust agent runtime 与 harness framework，用于构建安全、
可扩展、可控的 tool-calling agent，并对 coding-agent 工作负载提供一等支持。

molo 是库和框架，不是最终用户产品。它不提供默认 CLI 体验、模型托管、账号、
计费或产品策略。类似 Claude Code / Codex 的 CLI coding agent 应构建在 molo
之上；molo 为这类产品提供可复用的 kernel 与治理层。

## 核心原则

四条原则驱动本文档中的每一条边界：

- **Agent 负责决策。** agent kernel 拥有推理状态：读取上下文、请求模型、
  选择下一步、发起副作用请求、消化 observation、产出最终回答。
- **Harness 负责治理与执行。** harness 拥有 config、workspace、policy、
  sandbox、approval、audit、transcript。它执行 agent 请求的 effect，并把
  observation 回灌给 agent。
- **Core 保持小而稳定。** 最内层 crate 只承载 message / run / tool / effect
  契约，不引入重依赖。
- **coding 支持是一等但可选的能力。** coding workload 能力是独立层，绝不
  成为 agent loop 的隐藏职责。

## 分层

| 层 | 职责 | 典型内容 | 明确不包含 |
| --- | --- | --- | --- |
| `molo-core` | 层与层之间最小且稳定的契约 | `Message`、`ContentBlock`、`ToolSchema`、`RunRequest`、`RunOutput`、`RunContext`、`AgentAction`、`Observation`、`EffectRequest`、`Artifact`、`Provider` trait、基础错误与事件类型 | `reqwest`、`rmcp`、`tokio::process`、`jsonschema`、`schemars`、shell/git/filesystem 实现 |
| `molo-agent` | 通用 agent runtime | `AgentKernel`、`ReActAgent`、`TypedAgent`、`Memory`、`WindowMemory`、`StructuredValidator`、`RunLoop`、`ToolRegistry`、`ToolRoundExecutor` | 副作用执行、approval/sandbox 策略、产品 UX |
| `molo-harness` | 副作用治理与执行 | `Harness`、`EffectKind`、`EffectRequest`、`EffectOutput`、`RiskLevel`、`ApprovalBroker`、`SandboxPolicy`、`NetworkPolicy`、`AuditSink`、`TranscriptStore`、`CommandExecutor` trait、`Workspace` trait、`ArtifactStore` | 模型交互、prompt 组装、产品 UX |
| `molo-coding` | coding workload SDK | `LocalWorkspace`、`PatchApplier`、`GitInspector`、`RepoSearcher`、`RipgrepSearcher`、`ShellCommandExecutor`、`TestRunner`、`InstructionResolver`、`CodingContextProvider`、effect adapter | CLI/TUI、产品策略、绕过 harness 的直接副作用 |
| `molo-mcp` | MCP adapter | MCP client、MCP tool/effect adapter、server namespace、permission bridge、tool unloading | 成为 `molo-core` 的默认依赖 |
| `molo-skills` | skills 协议 | `Skill`、`SkillRegistry`、skill 加载、progressive disclosure、allowed tools | effect 执行、成为 `molo-core` 的默认依赖 |
| `molo-cli` | 用于验证架构的 reference CLI | `chat`、`code`、`review`、`resume` 命令、config、transcript、approval prompt | molo 的主要产品形态 |

## 依赖方向

依赖只能向下，任何层都不能依赖上层：

```text
                        molo-cli
                      /    |    \
            molo-coding    |     \
                 \    molo-harness  molo-mcp   molo-skills
                  \      |            \          /
                   \     |             \        /
                    molo-agent          \      /
                        \                \    /
                         \                v  /
                          +----- molo-core ----+
```

具体规则：

- `molo-core` 不依赖任何其他 molo crate，并保持第三方依赖最小。
- `molo-agent` 依赖 `molo-core`。
- `molo-harness` 依赖 `molo-agent` 与 `molo-core`：它在外层驱动 agent
  kernel，并执行其 effect request。
- `molo-coding` 依赖 `molo-harness` 与 `molo-core`：实现 workspace 与
  command 的 effect adapter，绝不依赖 `molo-agent`。
- `molo-mcp` 依赖 `molo-core`；存在 harness 时，其 permission bridge 把
  MCP 副作用转换为 harness effect。
- `molo-skills` 依赖 `molo-agent` 与 `molo-core`：是 loop 的装配层，不执行
  任何 effect。
- `molo-cli` 可以依赖所有层；任何层都不能依赖 `molo-cli`。

重依赖（`reqwest`、`rmcp`、`jsonschema`、`schemars`、shell/git/filesystem
机制）只属于真正需要它们的层，通过 feature flag 或 crate 边界隔离，绝不
引入 `molo-core`。

## 副作用执行链路

agent 自身绝不直接执行生产副作用。链路为：

```text
model -> tool/effect request -> risk classification -> policy check
      -> approval (if required) -> sandbox/executor -> output collection
      -> truncation/redaction -> audit/transcript -> observation -> model
```

推论：

- tool 可以返回普通 `ToolOutput`（纯计算或低风险工作），也可以返回
  `EffectRequest`；effect 由 harness 执行。
- 策略由代码强制，绝不只写在 prompt 里。prompt 可以描述策略，但不是安全
  边界。
- approval 是可插拔的 broker；sandbox 与 network 策略是显式且可测试的。
- 每个被执行的 effect 都会产出 agent 消化的 observation，以及 harness
  保留的 audit 记录。

## molo 不是什么

- 不是面向最终用户的 CLI coding agent 产品。
- 不是模型提供商或 API 网关。
- 不是分布式 workflow 编排器、agent OS 或长时任务调度器。
- 不是其运行所在的 OS / 容器 / sandbox 安全能力的替代品；molo 在其之上
  叠加策略与审计。

## 现状与目标

`0.2.x` 是单个 `molo` crate 加 `molo-macros`，交付的是 agent-runtime 一侧：
`ReActAgent` loop、provider、memory、tool registry、MCP、skills、structured
output、event 与 cancellation。harness 与 coding workload 层尚未成为独立
crate。

上述拆分是演进方向，将逐步抽取，而不是一次性完成。在 1.0 之前，当目标
架构需要时，public API 允许跨 0.x minor 破坏性变更，每次 breaking change
都附带迁移路径。治理细节——API 稳定性分级、发布规则与 harness 威胁模型——
由项目维护者随开发路线图一起维护。

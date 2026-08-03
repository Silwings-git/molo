# molo

**molo**(Model Loop)——Rust 编写的 AI coding agent 框架。

`molo` 是框架的**聚合入口 crate**:re-export [`molo-kernel`](https://github.com/Silwings-git/molo-kernel) 契约层与 [`molo-rt`](https://github.com/Silwings-git/molo-rt) 运行时,使用者只需声明 `molo = "0.1"` 一个依赖。

- `molo-kernel`:契约层——类型、trait、事件协议,零业务实现
- `molo-rt`:完整运行时——循环、工具注册表、策略引擎、provider、Skill、MCP、钩子、命令

## 状态

设计定稿(2026-08-03),实现进行中。当前版本为名称占位,re-export 随首个实现版本发布。

- 设计文档:[molo-kernel-design.md](https://gitee.com/silwings/my-coding-agent/blob/main/docs/molo-kernel-design.md) / [molo-rt-design.md](https://gitee.com/silwings/my-coding-agent/blob/main/docs/molo-rt-design.md)
- agent 产品(mca)独立于框架仓库开发

## License

MIT

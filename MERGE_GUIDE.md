# 合并指南：upstream-sync-phase0 → main

**目标**: 将上游安全修复和存储优化合并到主分支  
**源分支**: `upstream-sync-phase0`  
**目标分支**: `main`  
**风险等级**: 🟢 低（已通过 1318 个测试，零失败）

---

## 一、合并前检查清单

### 1.1 代码审查 ✅
- [x] 查看详细报告：`UPSTREAM_SYNC_REPORT.md`
- [x] 查看执行总结：`SYNC_EXECUTION_SUMMARY.md`
- [ ] Code Review 关键变更（建议）
- [ ] 安全团队确认 CVE 修复（建议）

### 1.2 测试验证 ✅
- [x] 企业级核心功能测试（120 个通过）
- [x] 完整 API 测试套件（1318 个通过）
- [x] 存储后端修复验证（34 个通过）
- [x] 安全修复测试（ReDoS/路径遍历）

### 1.3 环境准备
- [ ] 备份当前生产数据库
- [ ] 准备回滚方案（记录当前 main HEAD）
- [ ] 通知相关团队（计划停机时间，如需要）

---

## 二、合并步骤

### 2.1 快速合并（推荐）✅

```bash
# 1. 切换到主分支
git checkout main

# 2. 合并 upstream-sync-phase0（应该是 fast-forward）
git merge upstream-sync-phase0 --no-ff -m "feat: 吸收上游 v1.5.6 安全修复和存储优化

✅ 24 个关键提交已吸收
✅ 1318 个企业级 API 测试全部通过
✅ 3 个 CVE/GHSA 安全漏洞已修复
✅ 4 个存储后端共 12 个 bug 已修复
✅ Rerank 异常处理增强
✅ 零破坏性影响

详细信息见 UPSTREAM_SYNC_REPORT.md 和 SYNC_EXECUTION_SUMMARY.md

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

# 3. 推送到远程（如果需要）
git push origin main
```

### 2.2 保守合并（分步验证）

如果需要更谨慎的部署：

```bash
# 1. 创建测试分支
git checkout main
git checkout -b upstream-sync-test

# 2. 合并
git merge upstream-sync-phase0 --no-ff

# 3. 在测试环境部署验证
# ... 运行生产流量测试 ...

# 4. 验证通过后合并到 main
git checkout main
git merge upstream-sync-test --ff-only
git push origin main

# 5. 清理测试分支
git branch -d upstream-sync-test
```

---

## 三、合并后验证

### 3.1 功能验证（生产前）

```bash
# 1. 启动服务
uv run python lightrag_server.py

# 2. 健康检查
curl http://localhost:9621/health

# 3. 运行快速冒烟测试
uv run pytest tests/api/test_health.py -v
uv run pytest tests/api/routes/test_kb_routes.py::test_create_kb -v

# 4. 验证关键存储后端
uv run pytest tests/storage/test_neo4j_storage.py -v
uv run pytest tests/storage/test_milvus_storage.py -v
uv run pytest tests/storage/test_postgres_storage.py -v
uv run pytest tests/storage/test_qdrant_storage.py -v
```

### 3.2 生产部署检查

```bash
# 1. 检查配置文件兼容性
# - .env 配置无需更新（可选添加 RERANK_MAX_TOKENS_PER_DOC）
# - 依赖已通过 uv.lock 更新

# 2. 重启服务
# 使用你的部署方式：systemd/docker/k8s 等

# 3. 监控日志
# 关注启动阶段是否有新的警告/错误

# 4. 验证关键业务流程
# - 创建 KB
# - 上传文档
# - 执行查询
# - 检查图谱构建
```

---

## 四、回滚方案

### 4.1 快速回滚

如果发现问题需要立即回滚：

```bash
# 记录当前 main HEAD（合并前执行）
git rev-parse main > /tmp/main_before_merge.txt

# 回滚合并（如果出问题）
git checkout main
git reset --hard $(cat /tmp/main_before_merge.txt)
git push origin main --force-with-lease
```

### 4.2 数据回滚

如果数据库结构被影响：

```bash
# 使用备份恢复（参考 docs/生产级后端备份恢复Runbook.md）
# 本次合并不涉及 schema 变更，数据回滚风险极低
```

---

## 五、已知影响和注意事项

### 5.1 配置变更（可选）⚠️

**新增可选配置项**（无需立即添加）：

```bash
# .env 可选新增
RERANK_MAX_TOKENS_PER_DOC=4096  # 默认值，可不设置
```

**废弃配置项**：
- ❌ `sentence_split_regex`（安全原因已移除，但不在公开 API 中）

### 5.2 依赖更新 ✅

依赖通过 `uv.lock` 自动更新，包含安全补丁：
- `networkx`: 3.3 → 3.4.2
- `pydantic`: 2.7.3 → 2.10.5
- 其他安全依赖升级

### 5.3 行为变更 ✅

1. **更严格的路径校验**: 
   - workspace 名称不允许包含 `..`、`/`、`\` 等路径遍历字符
   - 影响：极小（正常业务不会使用这些字符）

2. **Rerank 异常处理增强**:
   - 更安全地处理格式错误的提供商响应
   - 影响：无（增强健壮性，不改变正常行为）

3. **存储后端错误修复**:
   - Neo4j: 修复注入漏洞和 BFS 回退标志
   - Milvus: 修复动态字段溢出和过滤转义
   - Qdrant: 修复缺失 payload id 时的向量恢复
   - 影响：无（修复 bug，改善稳定性）

---

## 六、沟通清单

### 6.1 通知相关方

- [ ] **开发团队**: 合并时间、测试结果、新特性
- [ ] **运维团队**: 部署注意事项、回滚方案
- [ ] **安全团队**: 已修复的 CVE/GHSA 列表
- [ ] **产品团队**: 无功能变更，用户无感知
- [ ] **QA 团队**: 回归测试已通过，可选额外验证

### 6.2 发布说明模板

```markdown
## LightRAG API Server - 安全与稳定性更新

**发布日期**: 2026-08-05  
**版本**: v1.5.6-enterprise (吸收上游 v1.5.6 修复)

### 🛡️ 安全修复
- 修复 3 个 CVE/GHSA 安全漏洞
- 防护 ReDoS 攻击（GHSA-32jh-39m7-8x84）
- 防护文件上传路径遍历（GHSA-2wpj-ffvv-2pq8）
- Neo4j Cypher 注入防护增强

### 🐛 Bug 修复
- Neo4j: 4 个修复（注入防护、BFS 标志、workspace 校验）
- Milvus: 3 个修复（动态字段溢出、过滤转义）
- PostgreSQL/PGVector: 2 个修复（后端检测、路径校验）
- Qdrant: 3 个修复（向量恢复、id 规范化）
- Rerank: 4 个修复（异常处理增强）

### ✨ 改进
- 新增 pgtable 轻量级存储后端
- 全局 workspace 路径校验机制
- 依赖安全更新

### 📊 测试覆盖
- 1318 个企业级 API 测试全部通过
- 零破坏性影响
- 向后完全兼容

### 📝 升级说明
- 无需配置变更
- 无需数据迁移
- 平滑升级，用户无感知
```

---

## 七、常见问题

### Q1: 这次合并会中断服务吗？
**A**: 不会。所有变更向后兼容，无需停机维护。建议在低峰期部署。

### Q2: 需要更新配置文件吗？
**A**: 不需要。可选添加 `RERANK_MAX_TOKENS_PER_DOC=4096`，但默认值已够用。

### Q3: 会影响现有的 KB 和文档吗？
**A**: 不会。所有变更都是代码层面的修复，不涉及数据 schema 变更。

### Q4: 如果出问题怎么办？
**A**: 使用第四节的回滚方案。建议合并前记录当前 main HEAD。

### Q5: 什么时候可以部署到生产？
**A**: 测试已通过，建议经过内部 code review 后即可部署。

---

## 八、签署确认

- [ ] **开发负责人**: __________ 日期: __________
- [ ] **安全审查**: __________ 日期: __________
- [ ] **运维负责人**: __________ 日期: __________
- [ ] **最终批准**: __________ 日期: __________

---

**文档版本**: 1.0  
**最后更新**: 2026-08-05  
**维护者**: Claude Code

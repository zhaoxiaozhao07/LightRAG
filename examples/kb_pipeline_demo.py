"""端到端 RAG 流程演示脚本。

运行示例：

    python examples/kb_pipeline_demo.py \
        --folder ./my_pdfs \
        --server http://localhost:9621 \
        --api-key <your_api_key> \
        --kb-id demo_kb \
        --question "文档主要讲了什么？"

第一次运行会：
    1. 创建 KB（若不存在）
    2. 把目录下所有支持的文件上传到 KB
    3. 等待 parse / build_kg 任务全部 succeeded
    4. 用 query 接口跑一次问答（不携带历史会话）

后续运行会：
    1. 读取目录旁的 .kb_state.json 状态文件
    2. 计算每个文件的 sha256
    3. 仅对新增 / 内容变化的文件触发上传 + 解析 + 构建
    4. 对未变化的文件直接跳过（hash 命中由服务端自动 skip）
    5. 检测到本地已删除的文件时仅打印告警（document delete 接口尚未实现）
    6. 再次运行问答

脚本依赖：``httpx``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover - 运行期检查
    raise SystemExit(
        "需要安装 httpx 才能运行该脚本：pip install httpx"
    ) from exc


# 与服务端 ``SUPPORTED_DOCUMENT_EXTENSIONS`` 保持一致的子集。可按需扩展。
DEFAULT_SUFFIXES = (
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".txt",
    ".md",
    ".html",
    ".htm",
)

STATE_FILENAME = ".kb_state.json"
PARSE_WAIT_TIMEOUT = 600.0
BUILD_WAIT_TIMEOUT = 1800.0
JOB_WAIT_SERVER_WINDOW = 600.0
JOB_WAIT_HTTP_GRACE = 10.0
JOB_WAIT_ACTIVE_STATES = {"queued", "running"}

QUERY_MODES = ("local", "global", "hybrid", "naive", "mix", "bypass")


def _print_help_banner() -> None:
    print(
        "\n"
        "================ 交互式问答 ================\n"
        " 直接输入问题回车提交（每次问答都不带历史会话）。\n"
        " 支持的指令：\n"
        f"   :mode <{'|'.join(QUERY_MODES)}>     切换检索模式\n"
        "   :refs on|off                        是否返回引用列表\n"
        "   :chunks on|off                      引用是否带 chunk 文本\n"
        "   :top_k <N> | :top_k clear           覆盖 top_k\n"
        "   :chunk_top_k <N> | :chunk_top_k clear   覆盖 chunk_top_k\n"
        "   :prompt <text> | :prompt clear      设置 / 清空 user_prompt\n"
        "   :show                               显示当前会话参数\n"
        "   :help                               显示本帮助\n"
        "   :quit / :exit / 空行 / Ctrl+D        退出\n"
        "============================================\n"
    )


def _print_query_params(params: dict[str, Any]) -> None:
    print(
        "[params] mode={mode} refs={refs} chunks={chunks} "
        "top_k={top_k} chunk_top_k={chunk_top_k} user_prompt={prompt}".format(
            mode=params["mode"],
            refs=params["include_references"],
            chunks=params["include_chunk_content"],
            top_k=params["top_k"] if params["top_k"] is not None else "<server default>",
            chunk_top_k=(
                params["chunk_top_k"]
                if params["chunk_top_k"] is not None
                else "<server default>"
            ),
            prompt=params["user_prompt"] or "<none>",
        )
    )


def _parse_bool_token(token: str) -> bool | None:
    lowered = token.strip().lower()
    if lowered in {"on", "true", "1", "yes", "y"}:
        return True
    if lowered in {"off", "false", "0", "no", "n"}:
        return False
    return None


def _handle_command(line: str, params: dict[str, Any]) -> bool:
    """处理 ``:command``；返回 True 表示已处理，False 表示视为问题。"""
    if not line.startswith(":"):
        return False
    parts = line.split(maxsplit=1)
    cmd = parts[0][1:].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in {"quit", "exit", "q"}:
        params["_quit"] = True
        return True
    if cmd in {"help", "h", "?"}:
        _print_help_banner()
        return True
    if cmd == "show":
        _print_query_params(params)
        return True
    if cmd == "mode":
        if arg not in QUERY_MODES:
            print(f"[warn] 无效模式：{arg!r}；支持 {QUERY_MODES}")
        else:
            params["mode"] = arg
            print(f"[ok] mode -> {arg}")
        return True
    if cmd == "refs":
        value = _parse_bool_token(arg)
        if value is None:
            print("[warn] 用法 :refs on|off")
        else:
            params["include_references"] = value
            print(f"[ok] include_references -> {value}")
        return True
    if cmd == "chunks":
        value = _parse_bool_token(arg)
        if value is None:
            print("[warn] 用法 :chunks on|off")
        else:
            params["include_chunk_content"] = value
            print(f"[ok] include_chunk_content -> {value}")
        return True
    if cmd == "top_k":
        if arg.lower() in {"clear", "reset", ""}:
            params["top_k"] = None
            print("[ok] top_k 已清空，使用服务端默认值")
        else:
            try:
                params["top_k"] = max(1, int(arg))
                print(f"[ok] top_k -> {params['top_k']}")
            except ValueError:
                print(f"[warn] :top_k 需要整数或 clear，收到 {arg!r}")
        return True
    if cmd == "chunk_top_k":
        if arg.lower() in {"clear", "reset", ""}:
            params["chunk_top_k"] = None
            print("[ok] chunk_top_k 已清空，使用服务端默认值")
        else:
            try:
                params["chunk_top_k"] = max(1, int(arg))
                print(f"[ok] chunk_top_k -> {params['chunk_top_k']}")
            except ValueError:
                print(f"[warn] :chunk_top_k 需要整数或 clear，收到 {arg!r}")
        return True
    if cmd == "prompt":
        if arg.lower() in {"clear", "reset", ""}:
            params["user_prompt"] = None
            print("[ok] user_prompt 已清空")
        else:
            params["user_prompt"] = arg
            print(f"[ok] user_prompt -> {arg!r}")
        return True
    print(f"[warn] 未知指令 :{cmd}（输入 :help 查看用法）")
    return True


def _print_answer(answer: dict[str, Any], *, params: dict[str, Any]) -> None:
    print("=" * 60)
    print(answer.get("response", ""))
    refs = answer.get("references") or []
    if params["include_references"] and refs:
        print("-" * 60)
        print("引用：")
        for ref in refs:
            print(f"  [{ref.get('reference_id', '?')}] {ref.get('file_path', '')}")
            if params["include_chunk_content"]:
                for chunk in ref.get("content") or []:
                    snippet = chunk.strip().replace("\n", " ")
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "..."
                    print(f"      · {snippet}")
    print("=" * 60)


def interactive_query_loop(
    client: "KBClient", kb_id: str, *, args: argparse.Namespace
) -> None:
    params: dict[str, Any] = {
        "mode": args.mode,
        "include_references": args.include_references,
        "include_chunk_content": False,
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
        "user_prompt": None,
        "_quit": False,
    }
    _print_help_banner()
    _print_query_params(params)
    turn = 0
    while True:
        try:
            line = input("\n[问答] ❯ ").strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print("\n[info] 用户中断，退出问答")
            return
        if not line:
            print("[info] 收到空行，退出问答")
            return
        if line.startswith(":"):
            _handle_command(line, params)
            if params["_quit"]:
                print("[info] 退出问答")
                return
            continue
        turn += 1
        print(f"[step] 第 {turn} 轮问答（每轮独立，不携带历史会话）")
        started = time.time()
        try:
            answer = client.query(
                kb_id,
                line,
                mode=params["mode"],
                include_references=params["include_references"],
                top_k=params["top_k"],
                chunk_top_k=params["chunk_top_k"],
                include_chunk_content=params["include_chunk_content"],
                user_prompt=params["user_prompt"],
            )
        except httpx.HTTPStatusError as exc:
            print(
                f"[error] HTTP {exc.response.status_code}: {exc.response.text}"
            )
            continue
        except httpx.HTTPError as exc:
            print(f"[error] 请求失败：{exc}")
            continue
        elapsed = time.time() - started
        _print_answer(answer, params=params)
        print(f"[info] 本轮耗时 {elapsed:.1f}s")


@dataclass(slots=True)
class FileEntry:
    """单个文件在状态文件中的快照。"""

    path: str
    content_hash: str
    document_id: str
    workspace_source_uri: str | None = None
    last_uploaded_at: str = ""


@dataclass(slots=True)
class PipelineState:
    """状态文件的完整结构。"""

    kb_id: str = ""
    files: dict[str, FileEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] 状态文件损坏，重新生成: {path}", file=sys.stderr)
            return cls()
        files = {
            key: FileEntry(**value)
            for key, value in (payload.get("files") or {}).items()
            if isinstance(value, dict)
        }
        return cls(kb_id=str(payload.get("kb_id") or ""), files=files)

    def save(self, path: Path) -> None:
        payload = {
            "kb_id": self.kb_id,
            "files": {
                key: {
                    "path": entry.path,
                    "content_hash": entry.content_hash,
                    "document_id": entry.document_id,
                    "workspace_source_uri": entry.workspace_source_uri,
                    "last_uploaded_at": entry.last_uploaded_at,
                }
                for key, entry in self.files.items()
            },
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightRAG KB 端到端流程演示（上传 -> 解析 -> KG 构建 -> 交互式问答）"
    )
    parser.add_argument("--folder", required=True, help="包含待入库文件的本地目录")
    parser.add_argument(
        "--server",
        default="http://localhost:9621",
        help="LightRAG API 服务地址",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="X-API-Key 头（无鉴权时留空）",
    )
    parser.add_argument(
        "--kb-id",
        required=True,
        help="目标知识库 id；不存在则自动创建",
    )
    parser.add_argument(
        "--kb-name",
        default=None,
        help="新建 KB 时使用的展示名称，缺省与 kb-id 相同",
    )
    parser.add_argument(
        "--mode",
        default="mix",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
        help="问答检索模式的初始值（交互过程中可用 :mode <name> 切换）",
    )
    parser.add_argument(
        "--include-references",
        action="store_true",
        help="问答时返回引用列表的初始值（交互过程中可用 :refs on/off 切换）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="top_k 初始值（交互过程中可用 :top_k <n> 切换；不传则使用服务端默认）",
    )
    parser.add_argument(
        "--chunk-top-k",
        type=int,
        default=None,
        help="chunk_top_k 初始值（交互过程中可用 :chunk_top_k <n> 切换）",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="自定义状态文件路径，默认放在 folder/.kb_state.json",
    )
    parser.add_argument(
        "--parser-engine",
        default=None,
        help="覆盖默认解析器（mineru / docling / native），缺省由服务端按扩展名自动选择",
    )
    parser.add_argument(
        "--process-options",
        default=None,
        help="解析阶段 process_options 字符串，如 'iF' / 'iteP'",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="只做上传 + 解析，不触发 build_kg（用于快速验证 parse）",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="跳过上传 / 解析 / 构建阶段，直接进入交互式问答",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="完成入库后不进入交互式问答，仅用于流程验证",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=BUILD_WAIT_TIMEOUT,
        help="单个 build_kg / parse job 的最长等待时间（秒）",
    )
    return parser.parse_args()


def discover_files(folder: Path) -> list[Path]:
    """枚举目录下所有支持的文件，按相对路径稳定排序。"""
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"目录不存在或不是文件夹: {folder}")
    files: list[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.name == STATE_FILENAME:
            continue
        if path.suffix.lower() in DEFAULT_SUFFIXES:
            files.append(path.resolve())
    files.sort()
    return files


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class KBClient:
    """对 LightRAG KB API 的极简同步封装。"""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0):
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    # KB ---------------------------------------------------------------
    def get_kb(self, kb_id: str) -> dict[str, Any] | None:
        response = self._client.get(f"/kbs/{kb_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def create_kb(self, kb_id: str, name: str) -> dict[str, Any]:
        response = self._client.post(
            "/kbs", json={"id": kb_id, "name": name or kb_id}
        )
        response.raise_for_status()
        return response.json()

    def ensure_kb(self, kb_id: str, name: str) -> dict[str, Any]:
        existing = self.get_kb(kb_id)
        if existing is not None:
            return existing
        return self.create_kb(kb_id, name)

    # Documents --------------------------------------------------------
    def upload_document(
        self,
        kb_id: str,
        file_path: Path,
        *,
        parser_engine: str | None = None,
        process_options: str | None = None,
    ) -> dict[str, Any]:
        # 上传仅写入 metadata，不触发解析；解析由后续显式 :parse 调用驱动，
        # 这样客户端可以稳定地等到 parse job 终态再决定下一步。
        params: dict[str, Any] = {"auto_parse": "false"}
        if parser_engine:
            params["parser_engine"] = parser_engine
        if process_options:
            params["process_options"] = process_options
        mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with file_path.open("rb") as fp:
            response = self._client.post(
                f"/kbs/{kb_id}/documents:upload",
                params=params,
                files={"files": (file_path.name, fp, mime)},
            )
        response.raise_for_status()
        return response.json()

    def parse_document(
        self,
        kb_id: str,
        document_id: str,
        *,
        parser_engine: str | None = None,
        process_options: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if parser_engine:
            body["engine"] = parser_engine
        if process_options:
            body["process_options"] = process_options
        response = self._client.post(
            f"/kbs/{kb_id}/documents/{document_id}:parse",
            json=body,
        )
        response.raise_for_status()
        return response.json()

    # Jobs -------------------------------------------------------------
    def wait_for_job(
        self, kb_id: str, job_id: str, *, timeout_seconds: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"job {job_id!r} did not reach a terminal state within "
                    f"{timeout_seconds:.1f}s"
                )
            wait_window = min(remaining, JOB_WAIT_SERVER_WINDOW)
            response = self._client.post(
                f"/kbs/{kb_id}/jobs/{job_id}:wait",
                params={"timeout_seconds": wait_window},
                timeout=wait_window + JOB_WAIT_HTTP_GRACE,
            )
            wait_timeout_detail = self._wait_timeout_detail(response)
            if wait_timeout_detail is not None:
                current_status = wait_timeout_detail.get("current_status")
                remaining_after_wait = deadline - time.monotonic()
                if current_status in JOB_WAIT_ACTIVE_STATES and remaining_after_wait > 0:
                    print(
                        f"[wait] job {job_id} 仍在 {current_status}，"
                        f"继续等待（剩余约 {remaining_after_wait:.0f}s）"
                    )
                    continue
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _wait_timeout_detail(response: httpx.Response) -> dict[str, Any] | None:
        if response.status_code != 408:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        detail = payload.get("detail")
        if not isinstance(detail, dict):
            return None
        if detail.get("error_code") != "wait_timeout":
            return None
        return detail

    # Build ------------------------------------------------------------
    def build_kg(self, kb_id: str, document_id: str) -> dict[str, Any]:
        response = self._client.post(
            f"/kbs/{kb_id}/documents/{document_id}:build-kg",
            json={},
        )
        response.raise_for_status()
        return response.json()

    # Query ------------------------------------------------------------
    def query(
        self,
        kb_id: str,
        question: str,
        *,
        mode: str,
        include_references: bool,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        include_chunk_content: bool = False,
        user_prompt: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": question,
            "mode": mode,
            "include_references": include_references,
            "include_chunk_content": include_chunk_content,
            "stream": False,
        }
        if top_k is not None:
            body["top_k"] = top_k
        if chunk_top_k is not None:
            body["chunk_top_k"] = chunk_top_k
        if user_prompt:
            body["user_prompt"] = user_prompt
        response = self._client.post(
            f"/kbs/{kb_id}/query",
            json=body,
            timeout=300.0,
        )
        response.raise_for_status()
        return response.json()


def run_pipeline(args: argparse.Namespace) -> int:
    folder = Path(args.folder).resolve()
    state_path = (
        Path(args.state_file).resolve()
        if args.state_file
        else folder / STATE_FILENAME
    )
    state = PipelineState.load(state_path)
    if state.kb_id and state.kb_id != args.kb_id:
        print(
            f"[warn] 状态文件记录的 kb_id={state.kb_id!r} 与当前 --kb-id={args.kb_id!r} 不一致，将按当前 kb 重新跟踪。"
        )
        state = PipelineState(kb_id=args.kb_id)
    state.kb_id = args.kb_id

    files = discover_files(folder)
    print(f"[info] 在 {folder} 下发现 {len(files)} 个候选文件")

    client = KBClient(args.server, args.api_key)
    try:
        kb_record = client.ensure_kb(args.kb_id, args.kb_name or args.kb_id)
        print(
            f"[info] 使用知识库 kb_id={kb_record['id']!r} workspace={kb_record['workspace']!r}"
        )

        if args.skip_ingest:
            print("[skip] --skip-ingest 已启用，跳过上传 / 解析 / 构建，直接进入问答")
            if not args.skip_query:
                interactive_query_loop(client, args.kb_id, args=args)
            return 0

        # 1. 计算 hash，分类：新增 / 变化 / 未变 / 已删
        new_files: list[Path] = []
        changed_files: list[Path] = []
        unchanged_files: list[Path] = []
        seen_keys: set[str] = set()

        for file_path in files:
            key = str(file_path.relative_to(folder)).replace("\\", "/")
            seen_keys.add(key)
            content_hash = hash_file(file_path)
            previous = state.files.get(key)
            if previous is None:
                new_files.append(file_path)
            elif previous.content_hash != content_hash:
                changed_files.append(file_path)
            else:
                unchanged_files.append(file_path)

        deleted_keys = sorted(set(state.files) - seen_keys)
        print(
            f"[info] 增量分析：新增 {len(new_files)} / 变化 {len(changed_files)} "
            f"/ 未变 {len(unchanged_files)} / 本地已删 {len(deleted_keys)}"
        )
        if deleted_keys:
            print(
                "[warn] 以下文件已不在目录中，但 LightRAG 当前还没有 KB 文档删除接口，"
                "对应的旧文档仍保留在知识库中，下一阶段补 DELETE / replace 时再处理："
            )
            for key in deleted_keys:
                print(f"        - {key} (doc_id={state.files[key].document_id})")

        to_upload: list[tuple[str, Path]] = [
            (str(f.relative_to(folder)).replace("\\", "/"), f)
            for f in (*new_files, *changed_files)
        ]

        # 2. 上传 + 解析
        upload_records: list[tuple[str, str]] = []  # (key, doc_id)
        for key, file_path in to_upload:
            print(f"[step] 上传：{key}")
            payload = client.upload_document(
                args.kb_id,
                file_path,
                parser_engine=args.parser_engine,
                process_options=args.process_options,
            )
            doc = payload["documents"][0]
            upload_records.append((key, doc["id"]))
            state.files[key] = FileEntry(
                path=key,
                content_hash=hash_file(file_path),
                document_id=doc["id"],
                workspace_source_uri=doc.get("source_uri"),
                last_uploaded_at=doc.get("created_at", ""),
            )

        if upload_records:
            print(f"[step] 触发解析：{len(upload_records)} 个文档")
            for key, doc_id in upload_records:
                parse_job = client.parse_document(
                    args.kb_id,
                    doc_id,
                    parser_engine=args.parser_engine,
                    process_options=args.process_options,
                )
                final = client.wait_for_job(
                    args.kb_id,
                    parse_job["id"],
                    timeout_seconds=min(args.timeout, PARSE_WAIT_TIMEOUT),
                )
                if final["status"] != "succeeded":
                    print(
                        f"[error] {key} 的解析失败：{final.get('error_code')} "
                        f"{final.get('error_message')}"
                    )
                else:
                    print(f"[ok]    {key} 解析成功 (job={parse_job['id']})")
        else:
            print("[skip] 没有需要重新上传 / 解析的文件")

        # 3. KG 构建（含 hash 命中自动 skip）
        if not args.no_build:
            build_targets: list[FileEntry] = list(state.files.values())
            print(f"[step] 触发 build_kg：{len(build_targets)} 个文档")
            for entry in build_targets:
                try:
                    response = client.build_kg(args.kb_id, entry.document_id)
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text
                    print(
                        f"[error] build_kg 失败：doc={entry.document_id} ({entry.path}) -> {detail}"
                    )
                    continue
                if response["status"] in {"succeeded", "failed"}:
                    skipped = (response.get("result") or {}).get("skipped")
                    suffix = " (skipped, hash matched)" if skipped else ""
                    print(
                        f"[ok]    build_kg {entry.path}: {response['status']}{suffix}"
                    )
                    continue
                # 仍在排队 / 运行中，等待终态
                final = client.wait_for_job(
                    args.kb_id, response["id"], timeout_seconds=args.timeout
                )
                if final["status"] != "succeeded":
                    print(
                        f"[error] build_kg {entry.path} 失败："
                        f"{final.get('error_code')} {final.get('error_message')}"
                    )
                else:
                    skipped = (final.get("result") or {}).get("skipped")
                    suffix = " (skipped, hash matched)" if skipped else ""
                    print(f"[ok]    build_kg {entry.path}: succeeded{suffix}")
        else:
            print("[skip] --no-build 已启用，跳过 KG 构建")

        # 4. 持久化状态文件，便于下次增量
        state.save(state_path)
        print(f"[info] 状态写回：{state_path}")

        # 5. 问答（交互式 while True，每轮独立，不带历史会话）
        if not args.skip_query:
            interactive_query_loop(client, args.kb_id, args=args)
        else:
            print("[skip] --skip-query 已启用，跳过问答")
        return 0
    finally:
        client.close()


def main() -> int:
    args = parse_args()
    started = time.time()
    try:
        code = run_pipeline(args)
    except httpx.HTTPStatusError as exc:
        print(
            f"[fatal] HTTP {exc.response.status_code} {exc.request.url}: "
            f"{exc.response.text}",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("[fatal] 用户中断", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    elapsed = time.time() - started
    print(f"[done] 全流程耗时 {elapsed:.1f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

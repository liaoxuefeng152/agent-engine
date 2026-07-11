"""
文件解析工具

提供多格式文件解析能力，Agent 可通过此工具读取和解析各类文件内容，
支持 CSV、Excel、PDF、TXT 等常见格式。用于数据分析、文档处理等场景。
"""
import os
import csv
import json
import logging
from typing import Optional

from langchain.tools import tool
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)

# 支持的文件格式
_SUPPORTED_EXTENSIONS = {
    ".csv", ".tsv",
    ".xlsx", ".xls",
    ".pdf",
    ".txt", ".md", ".json", ".jsonl",
}

# 文件大小限制 (10MB)
_MAX_FILE_SIZE = 10 * 1024 * 1024


def _parse_csv(file_path: str, max_rows: int = 200) -> str:
    """CSV/TSV 解析核心逻辑"""
    delimiter = "\t" if file_path.endswith(".tsv") else ","

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)

    if not rows:
        return "文件为空。"

    columns = list(rows[0].keys())
    total_info = f"共 {len(columns)} 列"

    lines = [f"文件解析结果 ({len(rows)} 行, {total_info}):"]
    lines.append(" | ".join(columns))
    lines.append("-" * 80)

    for row in rows:
        values = [str(v)[:50] for v in row.values()]
        lines.append(" | ".join(values))

    if len(rows) >= max_rows:
        lines.append(f"\n(已截断至 {max_rows} 行)")

    return "\n".join(lines)


def _parse_excel(file_path: str, max_rows: int = 200) -> str:
    """Excel 解析核心逻辑"""
    import openpyxl

    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    output_parts = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows_data = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= max_rows:
                break
            rows_data.append([str(cell) if cell is not None else "" for cell in row])

        if not rows_data:
            output_parts.append(f"\n[工作表: {sheet_name}] - 空表")
            continue

        columns = rows_data[0]
        data_rows = rows_data[1:]

        output_parts.append(f"\n[工作表: {sheet_name}] ({len(data_rows)} 行, {len(columns)} 列)")
        output_parts.append(" | ".join(columns))
        output_parts.append("-" * 80)

        for row in data_rows:
            values = [v[:50] for v in row]
            output_parts.append(" | ".join(values))

        if len(data_rows) >= max_rows:
            output_parts.append(f"(已截断至 {max_rows} 行)")

    wb.close()
    return "\n".join(output_parts)


def _parse_pdf(file_path: str, max_pages: int = 20) -> str:
    """PDF 解析核心逻辑"""
    import pdfplumber

    output_parts = []
    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        output_parts.append(f"PDF 文档: 共 {total_pages} 页")

        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                output_parts.append(f"\n(已截断至 {max_pages} 页)")
                break

            text = page.extract_text()
            if text and text.strip():
                output_parts.append(f"\n--- 第 {i + 1} 页 ---")
                output_parts.append(text.strip())

            # 尝试提取表格
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                if table:
                    output_parts.append(f"\n[表格 {j + 1}]")
                    for row in table:
                        cleaned = [str(cell)[:50] if cell else "" for cell in row]
                        output_parts.append(" | ".join(cleaned))

    return "\n".join(output_parts)


def _parse_text(file_path: str, max_chars: int = 50000) -> str:
    """纯文本/Markdown/JSON 解析核心逻辑"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read(max_chars)

    if len(content) >= max_chars:
        content += f"\n\n... (已截断至 {max_chars} 字符)"

    # 如果是 JSON，尝试格式化
    if file_path.endswith((".json",)):
        try:
            data = json.loads(content)
            content = json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass

    return content


@tool
def parse_file(file_path: str, max_rows: int = 200) -> str:
    """
    Parse and extract content from files. Supports CSV, Excel (.xlsx/.xls), PDF, TXT, Markdown, and JSON.
    Use this tool to read data files, analyze spreadsheets, extract text from PDFs, etc.

    Args:
        file_path: Path to the file to parse. Can be absolute or relative to the project root.
                  Supported formats: .csv, .tsv, .xlsx, .xls, .pdf, .txt, .md, .json, .jsonl
        max_rows: Maximum number of rows to read for tabular data (default: 200).
                  For text/PDF, this parameter is ignored.

    Returns:
        The extracted content from the file, formatted as readable text.
    """
    ctx = request_context.get() or new_context(method="parse_file")

    # Resolve file path
    workspace = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    if not os.path.isabs(file_path):
        file_path = os.path.join(workspace, file_path)

    # Check file exists
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"

    if not os.path.isfile(file_path):
        return f"路径不是文件: {file_path}"

    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > _MAX_FILE_SIZE:
        return f"文件过大 ({file_size / 1024 / 1024:.1f}MB)，超过限制 {_MAX_FILE_SIZE // 1024 // 1024}MB。"

    # Check extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        return f"不支持的文件格式: {ext}。支持: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"

    try:
        if ext in (".csv", ".tsv"):
            return _parse_csv(file_path, max_rows)
        elif ext in (".xlsx", ".xls"):
            return _parse_excel(file_path, max_rows)
        elif ext == ".pdf":
            return _parse_pdf(file_path)
        elif ext in (".txt", ".md", ".json", ".jsonl"):
            return _parse_text(file_path)
        else:
            return f"不支持的文件格式: {ext}"

    except Exception as e:
        logger.error(f"File parse failed: {e}")
        return f"文件解析失败: {e}"


@tool
def list_files(directory: str = "assets", pattern: str = "*") -> str:
    """
    List files in a directory, optionally filtered by a glob pattern.
    Use this tool to discover available files before parsing them.

    Args:
        directory: Directory path relative to project root (default: "assets").
        pattern: Glob pattern to filter files (default: "*" for all files).
                Examples: "*.csv", "*.xlsx", "data/*"

    Returns:
        A list of files with their sizes and modification times.
    """
    ctx = request_context.get() or new_context(method="list_files")

    workspace = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    if not os.path.isabs(directory):
        directory = os.path.join(workspace, directory)

    if not os.path.exists(directory):
        return f"目录不存在: {directory}"

    import glob as glob_module
    search_pattern = os.path.join(directory, "**", pattern)
    files = glob_module.glob(search_pattern, recursive=True)

    # Filter to files only
    files = [f for f in files if os.path.isfile(f)]

    if not files:
        return f"目录 {directory} 中没有匹配 '{pattern}' 的文件。"

    lines = [f"文件列表 ({len(files)} 个文件):"]
    for f in sorted(files):
        rel_path = os.path.relpath(f, workspace)
        size = os.path.getsize(f)
        size_str = f"{size / 1024:.1f}KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f}MB"

        import datetime
        mtime = os.path.getmtime(f)
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        _, ext = os.path.splitext(f)
        lines.append(f"  {rel_path}  ({size_str}, {mtime_str})")

    return "\n".join(lines)

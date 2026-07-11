"""
数据库只读查询工具

提供安全的数据库只读查询能力，Agent 可通过此工具查询业务数据，
如客户信息、订单状态、统计数据等。仅支持 SELECT 查询，禁止写操作。
"""
import logging
from typing import Optional

from langchain.tools import tool
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)

# 禁止执行的 SQL 关键字（安全防护）
_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
    "MERGE", "REPLACE", "CALL",
]

# 允许查询的表白名单（按需扩展）
_ALLOWED_TABLES = [
    "lark_agent_info",
    "lark_agent_chat",
    "lark_agent_token_stat",
    "lark_leads",
    "lark_customer",
    "lark_order",
]


def _validate_query(sql: str) -> tuple[bool, str]:
    """
    SQL 安全校验核心逻辑（可被多个 tool 复用）

    校验规则：
    1. 仅允许 SELECT 语句
    2. 禁止危险关键字
    3. 建议查询表在白名单内
    """
    sql_upper = sql.strip().upper()

    # 必须以 SELECT 开头
    if not sql_upper.startswith("SELECT"):
        return False, "仅支持 SELECT 查询语句，禁止写操作。"

    # 检查危险关键字
    for keyword in _FORBIDDEN_KEYWORDS:
        if keyword in sql_upper:
            return False, f"检测到禁止的 SQL 关键字: {keyword}"

    # 检查是否包含子查询中的写操作
    if any(kw in sql_upper for kw in ["INTO ", "SET ", "VALUES "]):
        return False, "检测到写操作子句，仅允许 SELECT 查询。"

    return True, ""


def _execute_read_query(sql: str, params: dict = None) -> list[dict]:
    """
    执行只读 SQL 查询核心逻辑（可被多个 tool 复用）
    """
    from storage.database.db import get_session
    from sqlalchemy import text

    with get_session() as session:
        result = session.execute(text(sql), params or {})
        columns = list(result.keys())
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]


@tool
def db_query(sql: str) -> str:
    """
    Execute a READ-ONLY SQL query against the business database.
    Use this tool to look up customer information, order status, statistics, agent configs, etc.
    Only SELECT statements are allowed. INSERT/UPDATE/DELETE/DROP are strictly forbidden.

    Args:
        sql: A valid PostgreSQL SELECT query. Example:
             SELECT agent_name, industry FROM lark_agent_info WHERE status = 1

    Returns:
        Query results as formatted text. Returns error if query is invalid or not a SELECT.
    """
    ctx = request_context.get() or new_context(method="db_query")

    # Validate SQL
    is_valid, error_msg = _validate_query(sql)
    if not is_valid:
        return f"SQL 校验失败: {error_msg}"

    # Limit result count
    sql_upper = sql.strip().upper()
    if "LIMIT" not in sql_upper:
        sql = sql.rstrip(";") + " LIMIT 100"

    try:
        results = _execute_read_query(sql)

        if not results:
            return "查询结果为空。"

        # Format as readable text
        columns = list(results[0].keys())
        total = len(results)

        # Build output
        lines = [f"查询结果 ({total} 条记录):"]
        lines.append(" | ".join(columns))
        lines.append("-" * 80)

        for row in results:
            values = [str(v) if v is not None else "NULL" for v in row.values()]
            lines.append(" | ".join(values))

        if total >= 100:
            lines.append("\n(结果已截断至100条，请添加更精确的过滤条件)")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"DB query failed: {e}")
        return f"数据库查询失败: {e}"


@tool
def get_table_schema(table_name: str) -> str:
    """
    Get the schema (column names, types, constraints) of a database table.
    Use this tool before writing queries to understand the table structure.

    Args:
        table_name: The name of the table to inspect.
                   Common tables: lark_agent_info, lark_agent_chat, lark_agent_token_stat

    Returns:
        Table schema information including column names and data types.
    """
    ctx = request_context.get() or new_context(method="get_table_schema")

    # Safety check
    if table_name.lower() not in _ALLOWED_TABLES:
        return f"表 {table_name} 不在允许查询的白名单中。允许的表: {', '.join(_ALLOWED_TABLES)}"

    try:
        sql = f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """
        results = _execute_read_query(sql)

        if not results:
            return f"表 {table_name} 不存在或无字段信息。"

        lines = [f"表 {table_name} 的结构:"]
        lines.append(f"{'字段名':<30} {'类型':<20} {'可空':<8} {'默认值'}")
        lines.append("-" * 80)

        for col in results:
            name = col.get("column_name", "")
            dtype = col.get("data_type", "")
            nullable = col.get("is_nullable", "")
            default = col.get("column_default", "") or ""
            lines.append(f"{name:<30} {dtype:<20} {nullable:<8} {default}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Get table schema failed: {e}")
        return f"获取表结构失败: {e}"

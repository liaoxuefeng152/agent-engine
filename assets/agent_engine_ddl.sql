-- ============================================
-- Agent-Engine 数据库表结构 DDL
-- 适配 PostgreSQL
-- ============================================

-- 1. 智能体配置表
CREATE TABLE IF NOT EXISTS lark_agent_info (
    id              BIGSERIAL PRIMARY KEY,
    agent_code      VARCHAR(64) NOT NULL UNIQUE,
    agent_name      VARCHAR(128) NOT NULL,
    industry        VARCHAR(64) NOT NULL,
    language        VARCHAR(16) NOT NULL DEFAULT 'zh',
    system_prompt   TEXT NOT NULL DEFAULT '',
    industry_rules  TEXT NOT NULL DEFAULT '',
    welcome_message TEXT NOT NULL DEFAULT '',
    model_config    JSONB NOT NULL DEFAULT '{}',
    knowledge_base_ids JSONB NOT NULL DEFAULT '[]',
    quota_config    JSONB NOT NULL DEFAULT '{}',
    nlu_config      JSONB NOT NULL DEFAULT '{}',
    status          INTEGER NOT NULL DEFAULT 1,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_industry ON lark_agent_info(industry);
CREATE INDEX IF NOT EXISTS idx_agent_status ON lark_agent_info(status);

COMMENT ON TABLE lark_agent_info IS '智能体配置表';
COMMENT ON COLUMN lark_agent_info.agent_code IS '智能体唯一编码';
COMMENT ON COLUMN lark_agent_info.model_config IS '模型路由配置: {"primary":"model_id","fallback":["model_id"],"timeout":30}';
COMMENT ON COLUMN lark_agent_info.knowledge_base_ids IS '绑定的知识库ID列表';
COMMENT ON COLUMN lark_agent_info.quota_config IS '配额配置: {"daily_limit":100,"per_agent_limit":50}';
COMMENT ON COLUMN lark_agent_info.nlu_config IS 'NLU意图关键词配置';

-- 2. 对话记录表
CREATE TABLE IF NOT EXISTS lark_agent_chat (
    id              BIGSERIAL PRIMARY KEY,
    chat_id         VARCHAR(64) NOT NULL UNIQUE,
    customer_id     VARCHAR(64) NOT NULL,
    agent_code      VARCHAR(64) NOT NULL,
    session_id      VARCHAR(64) NOT NULL,
    role            VARCHAR(16) NOT NULL,
    content         TEXT NOT NULL,
    message_type    VARCHAR(16) NOT NULL DEFAULT 'text',
    intent          VARCHAR(32),
    model_used      VARCHAR(64),
    token_input     INTEGER NOT NULL DEFAULT 0,
    token_output    INTEGER NOT NULL DEFAULT 0,
    source          VARCHAR(32) NOT NULL DEFAULT 'geo',
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_customer ON lark_agent_chat(customer_id);
CREATE INDEX IF NOT EXISTS idx_chat_agent ON lark_agent_chat(agent_code);
CREATE INDEX IF NOT EXISTS idx_chat_session ON lark_agent_chat(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_created ON lark_agent_chat(created_at);

COMMENT ON TABLE lark_agent_chat IS '对话记录表';
COMMENT ON COLUMN lark_agent_chat.role IS '角色: user/assistant/system';
COMMENT ON COLUMN lark_agent_chat.intent IS 'NLU识别意图: qa/lead_guide/wechat_invoke/human_transfer';
COMMENT ON COLUMN lark_agent_chat.source IS '来源: geo/wechat/website';

-- 3. Token消耗统计表
CREATE TABLE IF NOT EXISTS lark_agent_token_stat (
    id                  BIGSERIAL PRIMARY KEY,
    customer_id         VARCHAR(64) NOT NULL,
    agent_code          VARCHAR(64) NOT NULL,
    model_name          VARCHAR(64) NOT NULL,
    stat_date           DATE NOT NULL,
    total_input_tokens  BIGINT NOT NULL DEFAULT 0,
    total_output_tokens BIGINT NOT NULL DEFAULT 0,
    total_requests      INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stat_customer_date ON lark_agent_token_stat(customer_id, stat_date);
CREATE INDEX IF NOT EXISTS idx_stat_agent_date ON lark_agent_token_stat(agent_code, stat_date);
CREATE INDEX IF NOT EXISTS idx_stat_model_date ON lark_agent_token_stat(model_name, stat_date);

COMMENT ON TABLE lark_agent_token_stat IS 'Token消耗统计表(按天聚合)';

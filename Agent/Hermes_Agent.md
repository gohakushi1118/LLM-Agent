# Hermes Agent
![image](https://hackmd.io/_uploads/HyTEA8dabl.png)
## How to Start?
[GitHub Document](https://github.com/nousresearch/hermes-agent)
- 1. Install
```bash=
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```
- 2. Open
```
hermes
```
- 3. Seting your System

## Features
- 有記憶能力 => `memories folder`
    - 瞬時記憶
        - Context Window
    - 短期記憶
        - `MEMORY.md`
    - 長期記憶
        - `USER.md`
- 會變聰明
    - 「Learning Loop」 ~= Reinforcement Learning
- 模型可自行決定
    - openrouter
- 多平台能作使用
    - Discord, Telegram ...
- skills and tools
    - Skills: 把 tools 串接起來的說明書，給 Agent 了解用
    - Tools: 特定功能和腳本，負責實際的運作

## Learning Loop
1. Memory Flush
        
- 配置 (config.yaml):
```bash=
memory:
    memory_enabled: true
    user_profile_enabled: true
    memory_char_limit: 2200   
    user_char_limit: 1375  
    nudge_interval: 10 # 每 10 turns 提醒 agent 更新記憶
    flush_min_turns: 6 # 至少 6 turns 後才執行 flush
```

- 流程：
```
對話進行中
  ↓
每 10 turns → agent 被 nudge 考慮更新記憶
  ↓
若 turns >= 6 → 觸發 flush
  ↓
flush_memories: 將對話摘要寫入
~/.hermes/memories/MEMORY.md  ← 事件記憶
~/.hermes/memories/USER.md    ← 使用者偏好
```
> 技術限制：兩個記憶文件有字元上限，超出時 agent 會自動壓縮/合併舊記憶

2. Context Compression
- 配置
```bash=                  
context:
    engine: compressor
    compression:
    enabled: true
    threshold: 0.5 # context 用量超過 50% 觸發
    target_ratio: 0.2 # 壓縮到 20%
    protect_last_n: 20 # 保護最近 20 條訊息不壓縮
```

- 流程：
```
Context 使用量 > 50%
  ↓
auxiliary.compression model 讀取前段對話
  ↓
產生摘要 → 替換掉舊訊息 → Context 縮減至 20%
  ↓
最近 20 條訊息原樣保留
```                     
3. Skill Creation
```bash=
skills:
    creation_nudge_interval: 15  # 每 15 turns 提醒 agent 建立 skill
    template_vars: true
```
> Skills 儲存在 ~/.hermes/skills/，格式為 Markdown prompt templates
                  
- 流程：
```
完成一個複雜任務後
  ↓
agent 判斷此流程是否可複用
  ↓
將工作流程寫成 skill 文件
  ↓
下次類似任務 → 直接呼叫 skill 而非重新推理
```
4. Session Persistence + Checkpoints
- 配置
```
每次對話 → 儲存至 ~/.hermes/sessions/session_YYYYMMDD_*.json

重要節點 → 快照至 ~/.hermes/checkpoints/
```
> [!NOTE]CORE
> 話中每隔幾輪，把重要資訊從對話 context 壓縮寫入持久化記憶檔，下次對話時再載入這些記憶，讓 agent 跨會話持續學習累積知識

## Relevant Code
### 一、記憶和技能執行
```python
# run_agent.py:2867-2913
def _spawn_background_review(self, messages_snapshot, review_memory=False, review_skills=False):
    import threading

    # 依觸發種類選 prompt
    if review_memory and review_skills:
        prompt = self._COMBINED_REVIEW_PROMPT
    elif review_memory:
        prompt = self._MEMORY_REVIEW_PROMPT
    else:
        prompt = self._SKILL_REVIEW_PROMPT

    def _run_review():
        # Fork 一個新的 AIAgent
        review_agent = AIAgent(
            model=self.model,
            max_iterations=8,
            quiet_mode=True,
            platform=self.platform,
            provider=self.provider,
        )
        # 共享記憶 store
        review_agent._memory_store = self._memory_store
        review_agent._memory_nudge_interval = 0
        review_agent._skill_nudge_interval = 0

        # 把對話 snapshot 當 history 餵給 review agent，叫它決定要不要存
        review_agent.run_conversation(
            user_message=prompt,
            conversation_history=messages_snapshot,
        )
```
```python
# run_agent.py:2832-2865
_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work "
    "style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. "
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)

_SKILL_REVIEW_PROMPT = (
    "Review the conversation above and consider saving or updating a skill if appropriate.\n\n"
    "Focus on: was a non-trivial approach used to complete a task that required trial "
    "and error, or changing course due to experiential findings along the way, ...\n\n"
    "If a relevant skill already exists, update it with what you learned. "
    "Otherwise, create a new skill if the approach is reusable.\n"
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)
```

### 二、flush_memories

```python
# run_agent.py:7360-7394
def flush_memories(self, messages=None, min_turns=None):
    """Give the model one turn to persist memories before context is lost.

    Called before compression, session reset, or CLI exit.
    """
    if self._memory_flush_min_turns == 0 and min_turns is None:
        return
    if "memory" not in self.valid_tool_names or not self._memory_store:
        return
    effective_min = min_turns if min_turns is not None else self._memory_flush_min_turns
    if self._user_turn_count < effective_min:
        return

    flush_content = (
        "[System: The session is being compressed. "
        "Save anything worth remembering — prioritize user preferences, "
        "corrections, and recurring patterns over task-specific details.]"
    )
    _sentinel = f"__flush_{id(self)}_{time.monotonic()}"
    flush_msg = {"role": "user", "content": flush_content, "_flush_sentinel": _sentinel}
    messages.append(flush_msg)

    response = _call_llm(
        task="flush_memories",
        messages=api_messages,
        tools=[memory_tool_def],
        temperature=_flush_temperature,
        max_tokens=5160,
    )
```
```python
# run_agent.py:7568-7578
def _compress_context(self, messages, system_message, ...):
    # Pre-compression memory flush: 在壓縮前讓 model 搶救記憶
    self.flush_memories(messages, min_turns=0)   # min_turns=0 表示強制執行

    # Notify external memory provider before compression discards context
    if self._memory_manager:
        self._memory_manager.on_pre_compress(messages)

    compressed = self.context_compressor.compress(messages, ...)
```

---

### 三、Context Compression
```python
# agent/context_compressor.py:1106-1181
def compress(self, messages, current_tokens=None, focus_topic=None):
    """
    Algorithm:
      1. Prune old tool results (cheap pre-pass, no LLM call)
      2. Protect head messages (system prompt + first exchange)
      3. Find tail boundary by token budget (~20K tokens of recent context)
      4. Summarize middle turns with structured LLM prompt
      5. On re-compression, iteratively update the previous summary
    """
    # Phase 1: 先把舊的工具結果刪掉
    messages, pruned_count = self._prune_old_tool_results(
        messages, protect_tail_count=self.protect_last_n,
        protect_tail_tokens=self.tail_token_budget,
    )

    # Phase 2: 算出頭尾保留區間
    compress_start = self.protect_first_n
    compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
    turns_to_summarize = messages[compress_start:compress_end]

    # Phase 3: 呼叫 LLM 產生結構化摘要
    summary = self._generate_summary(turns_to_summarize, focus_topic=focus_topic)

    # Phase 4: 組合 = [head] + [summary] + [tail]
    compressed = [...head..., summary_msg, ...tail...]
    return compressed
```

## Skills
- autonomous-ai-agents
    - hermes-agent: Hermes Agent 完整指南
    - claude-code: Anthropic 程式任務
    - codex: Codex 程式任務
    - opencode: OpenCode 程式任務
- creative
    - architecture-diagram：生成 SVG 圖表
    - ascii-art：生成 ASCII 藝術
    - ascii-video：將任何格式轉換為 ASCII 藝術影片
    - baoyu-comic：生成知識教育漫畫
    - baoyu-infographic：生成專業資訊圖表
    - excalidraw：創建手繪風格圖表
    - ideation：透過創意限制生成專案點子
    - manim-video：使用 Manim 生成數學/技術動畫
    - p5js：互動式/生成式視覺藝術
    - pixel-art：將圖片轉換為復古像素藝術
    - popular-web-designs：網站的設計系統模板
    - songwriting-and-ai-music：歌曲創作與 AI 音樂生成技巧
- data-science
    - jupyter-live-kernel：使用 Jupyter kernel 進行執行
- devops
    - 管理 webhook 訂閱
- diagramming
    - architecture-diagram - 深色主題軟體架構圖 (HTML+SVG)
    - excalidraw - 手繪風格圖表
    - manim-video - 數學/技術動畫影片
    - p5js - 生成式視覺藝術
- dogfood
    - 系統性網頁 QA 測試
- email
    - himalaya：基於 IMAP/SMTP 的郵局管理工具
- gaming
    - minecraft-modpack-server：設定模組化 Minecraft 伺服器
    - pokemon-player：自動玩 Pokemon 遊戲
- github
    - codebase-inspection：分析 GitHub 帳號
    - github-auth：設定 GitHub 認證
    - github-code-review：分析 git diff 並留下 PR 內聯評論
    - github-issues：創建/管理/搜尋 GitHub Issues
    - github-pr-workflow：完整的 Pull Request 生命週期
    - github-repo-management：下載/創建/複製/管理 GitHub 倉庫
- mcp
    - native-mcp：內建 MCP 客戶端，連接外部 MCP 服務器並自動註冊工具
- media
    - gif-search：從 Tenor 搜尋和下載 GIF
    - heartmula：運行 HeartMuLa 開源音樂生成模型
    - songsee：生成音訊光譜圖和音訊特徵可視化
    - youtube-content：抓取 YouTube 轉錄並轉換為結構化內容
- mlops
    - audiocraft-audio-generation：MusicGen/AudioGen 音訊生成
    - dspy：使用 DSPy 構建複雜 AI 系統
    - evaluating-llms-harness：在學術基準上評估 LLM
    - fine-tuning-with-trl：使用強化學習微調 LLM
    - huggingface-hub：搜尋/下載/上傳 HuggingFace 模型
    - llama-cpp：本地 GGUF 推理
    - obliteratus：移除 LLM 拒絕行為
    - outlines：保證生成結構化輸出
    - segment-anything-model：圖像分割基礎模型
    - serving-llms-vllm：使用 vLLM 部署高吞吐 LLM 服務
    - unsloth：快速微調
    - weights-and-biases：MLOps 實驗追蹤平台
    - axolotl：使用 Axolotl 微調 LLM
- note-taking
    - obsidian：讀取/搜尋/創建 Obsidian 筆記
- productivity
    - google-workspace：Gmail/Calendar/Drive/Sheets/Docs 整合
    - linear：管理 Linear 問題/專案/團隊
    - maps：位置智能
    - nano-pdf：用自然語言編輯 PDF
    - notion：管理 Notion 頁面/資料庫
    - ocr-and-documents：從 PDF/掃描文件提取文字
    - powerpoint：任何 .pptx 文件處理 (創建/讀取/修改)
- red-teaming
    - godmode：使用 G0DM0D3 技術越獄 API 服務的 LLM
- research
    - arxiv：搜尋和檢索 arXiv 學術論文
    - blogwatcher：監控部落格和 RSS/Atom 訂閱源
    - llm-wiki：建立和維護 LLM 維基知識庫
    - polymarket：查詢 Polymarket 預測市場數據
    - research-paper-writing：ML/AI 研究論文撰寫端到端流程
- smart-home
    - openhue：控制 Philips Hue 燈光、房間和場景
- social-media
    - xurl：與 X/Twitter 互動
- software-development
    - plan：寫出 Markdown 計劃但不執行
    - requesting-code-review：提交前驗證管線
    - subagent-driven-development：使用獨立子代理執行實現計劃

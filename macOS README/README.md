# Multi AI Agents Tool 🧠
###### !This is the macOS guide!

> A simple, tool for macOS — inspired by Pyx.Inc (https://github.com/Pyx.Inc)

M.A.A.T lets different AI models, work together, and chat with you, you can give them tasks and they will do it for you!

----

## Installation

1. Download the latest release from "https://github.com/M-lanGH/Multi-AI-Agent-Tool"
2. Open Terminal and check wich version:

```bash
pip3 --version
```
3. If that didn't work or you want to update to make sure, type:

```bash
python3 -m pip install --upgrade pip
```

4. If it gives you a warning about a newer version just copy and paste what it gives you, Example:

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 -m pip install --upgrade pip
```

6. After that you type:
   
```bash
cd Downloads
```

6. Then run the script:
   
```bash
python3 multi_agent-*version you have*.py
```

7. If it doesn't work try: (!type the commands below 1 by 1!)

```bash
cd Downloads
ls
```

8. 

Do you see the "multi_agent-*version you have*.py
then there is a problem with the Terminal, if you don't see the file then it isn't downloaded, download it by going to "https://github.com/M-lanGH/Multi-AI-Agent-Tool"

9. After you downloaded the file try it again by typing: (!type the commands below 1 by 1!)

```bash
cd Downloads
ls
```

10. If you see it there, then type in:

```bash
python3 multi_agent-*version you have downloaded*.py
```

11. Now it should work.




----





## Download The Models:

### Ollama:

#### ── Llama (Meta) ──────────────────────────────────────
ollama pull llama4                    # Latest Meta multimodal MoE model
ollama pull llama3.3                  # State of the art 70B
ollama pull llama3.2                  # 1B and 3B small models
ollama pull llama3.2-vision           # Vision 11B and 90B
ollama pull llama3.1                  # 8B, 70B, 405B
ollama pull llama3                    # 8B, 70B
ollama pull llama3-chatqa             # NVIDIA fine-tune for Q&A / RAG
ollama pull llama3-gradient           # Extended 1M context window
ollama pull llama3-groq-tool-use      # Groq tool use / function calling
ollama pull llama3-guard3             # Safety classification
ollama pull llama2                    # 7B, 13B, 70B
ollama pull llama2-uncensored         # Uncensored 7B, 70B
ollama pull llama2-chinese            # Chinese fine-tune 7B, 13B
ollama pull llama-pro                 # Programming & math specialist
ollama pull llama-guard3              # Content safety 1B, 8B

#### ── Mistral / Mixtral ────────────────────────────────
ollama pull mistral                   # 7B flagship
ollama pull mistral-nemo              # 12B with 128K context
ollama pull mistral-small             # 22B-24B benchmark setter
ollama pull mistral-small3.1          # Vision + 128K context 24B
ollama pull mistral-small3.2          # Improved function calling 24B
ollama pull mistral-large             # 123B flagship
ollama pull mistral-large-3           # Multimodal MoE (cloud)
ollama pull mistral-medium-3.5        # 128B vision+reasoning+code
ollama pull mistral-openorca          # OpenOrca fine-tune 7B
ollama pull mistrallite               # Long context fine-tune 7B
ollama pull mixtral                   # MoE 8x7B and 8x22B
ollama pull mistral-openorca          # OpenOrca 7B
ollama pull magistral                 # Reasoning 24B
ollama pull mathstral                 # Math reasoning 7B
ollama pull codestral                 # Code generation 22B
ollama pull devstral                  # Agentic coding 24B
ollama pull devstral-small-2          # Agentic coding vision 24B
ollama pull devstral-2                # 123B coding agent (cloud)
ollama pull ministral-3               # Edge deployment 3B, 8B, 14B

#### ── Gemma (Google) ───────────────────────────────────
ollama pull gemma4                    # Frontier vision+tools 26B, 31B
ollama pull gemma3                    # 270M, 1B, 4B, 12B, 27B
ollama pull gemma3n                   # On-device e2b, e4b
ollama pull gemma2                    # 2B, 9B, 27B
ollama pull gemma                     # Original 2B, 7B
ollama pull codegemma                 # Code 2B, 7B
ollama pull shieldgemma               # Safety eval 2B, 9B, 27B
ollama pull embeddinggemma            # Embeddings 300M
ollama pull translategemma            # Translation 55 languages 4B-27B
ollama pull medgemma                  # Medical vision 4B, 27B
ollama pull medgemma1.5               # Updated medical 4B
ollama pull gemini-3-flash-preview    # Gemini 3 Flash (cloud)
ollama pull functiongemma             # Function calling 270M

#### ── Qwen (Alibaba) ───────────────────────────────────
ollama pull qwen3                     # Latest 0.6B-235B dense+MoE
ollama pull qwen3.5                   # Multimodal 0.8B-122B
ollama pull qwen3.6                   # Coding 27B, 35B
ollama pull qwen3-coder               # Agentic coding 30B, 480B
ollama pull qwen3-coder-next          # Local agentic coding
ollama pull qwen3-vl                  # Vision-language 2B-235B
ollama pull qwen3-embedding           # Text embeddings 0.6B-8B
ollama pull qwen3-next                # Efficient reasoning 80B (cloud)
ollama pull qwen2.5                   # 0.5B-72B multilingual
ollama pull qwen2.5-coder             # Code 0.5B-32B
ollama pull qwen2.5vl                 # Vision-language 3B-72B
ollama pull qwen2                     # 0.5B-72B
ollama pull qwen2-math                # Math 1.5B-72B
ollama pull qwen                      # Qwen 1.5 0.5B-110B
ollama pull qwq                       # Reasoning 32B
ollama pull codeqwen                  # Code 7B

#### ── DeepSeek ─────────────────────────────────────────
ollama pull deepseek-r1               # Reasoning 1.5B-671B
ollama pull deepseek-v3               # MoE 671B
ollama pull deepseek-v3.1             # Hybrid thinking 671B (cloud)
ollama pull deepseek-v3.2             # Reasoning+agent (cloud)
ollama pull deepseek-v4-flash         # Preview 284B MoE (cloud)
ollama pull deepseek-v4-pro           # Frontier 1M context (cloud)
ollama pull deepseek-v2               # MoE 16B, 236B
ollama pull deepseek-v2.5             # Chat+code 236B
ollama pull deepseek-coder            # Code 1.3B-33B
ollama pull deepseek-coder-v2         # Code MoE 16B, 236B
ollama pull deepseek-llm              # 7B, 67B
ollama pull deepseek-ocr              # OCR vision 3B
ollama pull deepscaler                # Math reasoning 1.5B
ollama pull deepcoder                 # Open coding 1.5B, 14B
ollama pull r1-1776                   # DeepSeek-R1 unbiased (Perplexity)

#### ── Phi (Microsoft) ──────────────────────────────────
ollama pull phi4                      # 14B STEM reasoning
ollama pull phi4-mini                 # 3.8B multilingual+tools
ollama pull phi4-reasoning            # 14B reasoning
ollama pull phi4-mini-reasoning       # 3.8B lightweight reasoning
ollama pull phi3                      # 3.8B, 14B
ollama pull phi3.5                    # 3.8B lightweight
ollama pull phi                       # Phi-2 2.7B

#### ── IBM Granite ──────────────────────────────────────
ollama pull granite4                  # 350M-3B tools
ollama pull granite4.1                # Enterprise 3B-30B tools
ollama pull granite3.3                # Reasoning 2B, 8B
ollama pull granite3.2                # Thinking 2B, 8B
ollama pull granite3.2-vision         # Document vision 2B
ollama pull granite3.1-dense          # Dense 2B, 8B
ollama pull granite3.1-moe            # MoE 1B, 3B
ollama pull granite3-dense            # Tool-use 2B, 8B
ollama pull granite3-moe              # MoE 1B, 3B
ollama pull granite3-guardian         # Risk detection 2B, 8B
ollama pull granite-code              # Code 3B-34B
ollama pull granite-embedding         # Embeddings 30M, 278M

#### ── Cohere ───────────────────────────────────────────
ollama pull command-r                 # Conversational 35B
ollama pull command-r-plus            # Enterprise 104B
ollama pull command-r7b               # Fast 7B
ollama pull command-r7b-arabic        # Arabic 7B
ollama pull command-a                 # Enterprise 111B
ollama pull aya                       # Multilingual 23 languages 8B, 35B
ollama pull aya-expanse               # Multilingual tools 8B, 32B

#### ── NVIDIA Nemotron ──────────────────────────────────
ollama pull nemotron                  # Llama 3.1 fine-tune 70B
ollama pull nemotron-mini             # Tools roleplay 4B
ollama pull nemotron3                 # Multimodal video+audio 33B
ollama pull nemotron-3-super          # MoE 120B (cloud)
ollama pull nemotron-3-nano           # Agentic 4B, 30B (cloud)
ollama pull nemotron-cascade-2        # MoE reasoning 30B

#### ── Kimi (Moonshot AI) ───────────────────────────────
ollama pull kimi-k2                   # MoE coding (cloud)
ollama pull kimi-k2-thinking          # Thinking MoE (cloud)
ollama pull kimi-k2.5                 # Multimodal agentic (cloud)
ollama pull kimi-k2.6                 # Coding+vision agentic (cloud)

#### ── GLM (Zhipu AI / Z.ai) ────────────────────────────
ollama pull glm4                      # Multilingual 9B
ollama pull glm-4.6                   # Reasoning+coding (cloud)
ollama pull glm-4.7                   # Coding (cloud)
ollama pull glm-4.7-flash             # Lightweight 30B (cloud)
ollama pull glm-5                     # Agentic 744B MoE (cloud)
ollama pull glm-5.1                   # Coding flagship (cloud)
ollama pull glm-ocr                   # OCR vision

#### ── MiniMax ──────────────────────────────────────────
ollama pull minimax-m2                # Coding+agentic (cloud)
ollama pull minimax-m2.1              # Multilingual (cloud)
ollama pull minimax-m2.5              # Productivity+coding (cloud)
ollama pull minimax-m2.7              # Professional (cloud)

#### ── LG AI Research (EXAONE) ──────────────────────────
ollama pull exaone3.5                 # Bilingual EN+KR 2.4B-32B
ollama pull exaone-deep               # Reasoning 2.4B-32B

#### ── Vision / Multimodal ──────────────────────────────
ollama pull llava                     # Vision 7B, 13B, 34B
ollama pull llava-llama3              # LLaVA + Llama 3 8B
ollama pull llava-phi3                # LLaVA + Phi3 3.8B
ollama pull bakllava                  # Mistral + LLaVA 7B
ollama pull moondream                 # Edge vision 1.8B
ollama pull minicpm-v                 # Vision 8B

#### ── Code Specialists ─────────────────────────────────
ollama pull codellama                 # Meta code 7B-70B
ollama pull starcoder2                # Code 3B, 7B, 15B
ollama pull starcoder                 # Code 80+ languages 1B-15B
ollama pull stable-code               # Code completion 3B
ollama pull magicoder                 # Code 7B
ollama pull phind-codellama           # Code 34B
ollama pull wizardcoder               # Code 33B
ollama pull dolphincoder              # Uncensored code 7B, 15B
ollama pull codegeex4                 # Code 9B
ollama pull yi-coder                  # Code 1.5B, 9B
ollama pull opencoder                 # Code EN+ZH 1.5B, 8B
ollama pull codebooga                 # Merged code 34B

#### ── Embeddings ───────────────────────────────────────
ollama pull nomic-embed-text          # Popular embeddings
ollama pull nomic-embed-text-v2-moe   # Multilingual MoE embeddings
ollama pull mxbai-embed-large         # State-of-the-art 335M
ollama pull all-minilm                # Compact 22M, 33M
ollama pull snowflake-arctic-embed    # Performance 22M-335M
ollama pull snowflake-arctic-embed2   # Multilingual 568M
ollama pull bge-m3                    # Multi-lingual 567M
ollama pull bge-large                 # BAAI 335M
ollama pull paraphrase-multilingual   # Clustering/search 278M

#### ── Reasoning ────────────────────────────────────────
ollama pull cogito                    # Hybrid reasoning 3B-70B
ollama pull cogito-2.1                # MIT license (cloud)
ollama pull openthinker               # Open reasoning 7B, 32B
ollama pull deepscaler                # Math reasoning 1.5B
ollama pull marco-o1                  # Alibaba reasoning 7B
ollama pull tulu3                     # Allen AI 8B, 70B
ollama pull smallthinker              # Qwen 2.5 reasoning 3B
ollama pull phi4-reasoning            # Microsoft 14B
ollama pull qwq                       # Qwen reasoning 32B
ollama pull lfm2                      # Hybrid on-device 24B
ollama pull lfm2.5-thinking           # Hybrid on-device 1.2B
ollama pull olmo-3                    # Open reasoning 7B, 32B
ollama pull olmo-3.1                  # Open tools 32B
ollama pull rnj-1                     # Essential AI code+STEM 8B (cloud)
ollama pull laguna-xs.2               # Local agentic coding MoE 33B

#### ── Uncensored / Fine-tunes ──────────────────────────
ollama pull dolphin3                  # General purpose 8B
ollama pull dolphin-llama3            # Uncensored 8B, 70B
ollama pull dolphin-mixtral           # Uncensored 8x7B, 8x22B
ollama pull dolphin-mistral           # Uncensored 7B
ollama pull dolphin-phi               # Uncensored 2.7B
ollama pull tinydolphin               # Uncensored 1.1B
ollama pull nous-hermes2              # Nous Research 10.7B, 34B
ollama pull nous-hermes2-mixtral      # Nous + Mixtral 8x7B
ollama pull hermes3                   # Nous latest 3B-405B
ollama pull openhermes                # Mistral fine-tune 7B
ollama pull nous-hermes               # General 7B, 13B
ollama pull wizard-vicuna-uncensored  # Uncensored 7B-30B
ollama pull wizardlm-uncensored       # Uncensored 13B
ollama pull llama2-uncensored         # Uncensored 7B, 70B
ollama pull everythinglm              # Uncensored 16K context 13B
ollama pull megadolphin               # Merged 120B
ollama pull reflection                # Reflection-tuning 70B
ollama pull athene-v2                 # Code+math+log 72B

#### ── Multilingual ─────────────────────────────────────
ollama pull sailor2                   # South-East Asia 1B-20B
ollama pull aya                       # 23 languages 8B, 35B
ollama pull aya-expanse               # 23 languages tools 8B, 32B
ollama pull llama2-chinese            # Chinese 7B, 13B
ollama pull yi                        # Bilingual 6B-34B
ollama pull internlm2                 # Chinese+EN 1.8B-20B

#### ── Classic / General Chat ───────────────────────────
ollama pull vicuna                    # Chat 7B-33B
ollama pull orca-mini                 # General 3B-70B
ollama pull orca2                     # Microsoft reasoning 7B, 13B
ollama pull openchat                  # ChatGPT-surpassing 7B
ollama pull zephyr                    # Helpful assistant 7B, 141B
ollama pull neural-chat               # Intel fine-tune 7B
ollama pull starling-lm               # RLHF chat 7B
ollama pull solar                     # Compact chat 10.7B
ollama pull solar-pro                 # Advanced 22B
ollama pull stablelm2                 # Multilingual 1.6B, 12B
ollama pull stablelm-zephyr           # Lightweight 3B
ollama pull stable-beluga             # Orca-style Llama 2 7B-70B
ollama pull samantha-mistral          # Philosophy/psychology 7B
ollama pull tinyllama                 # Ultra compact 1.1B
ollama pull smollm2                   # HuggingFace 135M-1.7B
ollama pull smollm                    # Compact 135M-1.7B
ollama pull olmo2                     # Open 7B, 13B
ollama pull falcon                    # TII 7B-180B
ollama pull falcon2                   # TII 11B
ollama pull falcon3                   # Efficient 1B-10B

#### ── Specialized ──────────────────────────────────────
ollama pull sqlcoder                  # SQL generation 7B, 15B
ollama pull duckdb-nsql               # Text-to-SQL 7B
ollama pull meditron                  # Medical Llama 2 7B, 70B
ollama pull medllama2                 # Medical Q&A 7B
ollama pull wizard-math               # Math 7B-70B
ollama pull nuextract                 # Info extraction 3.8B
ollama pull reader-lm                 # HTML to Markdown 0.5B, 1.5B
ollama pull bespoke-minicheck         # Fact-checking 7B
ollama pull llama-guard3              # Safety 1B, 8B
ollama pull shieldgemma               # Safety eval 2B-27B
ollama pull nexusraven                # Function calling 13B
ollama pull firefunction-v2           # Function calling 70B
ollama pull dbrx                      # Databricks 132B
ollama pull deepseek-ocr              # OCR 3B
ollama pull glm-ocr                   # Document OCR
ollama pull codegeex4                 # Code 9B

#### ── Other / Community Highlights ─────────────────────
ollama pull wizardlm2                 # Microsoft 7B, 8x22B
ollama pull wizardlm                  # General Llama 2
ollama pull wizard-vicuna             # Chat 13B
ollama pull xwinlm                    # Llama 2 fine-tune 7B, 13B
ollama pull open-orca-platypus2       # Merged chat+code 13B
ollama pull yarn-llama2               # 128K context 7B, 13B
ollama pull yarn-mistral              # 64K-128K context 7B
ollama pull goliath                   # Merged Llama 2 70B
ollama pull alfred                    # Chat+instruct 40B
ollama pull notus                     # Zephyr fine-tune 7B
ollama pull notux                     # MoE fine-tune 8x7B
ollama pull codeup                    # Llama 2 code 13B
ollama pull megadolphin               # Merged 120B
ollama pull mistrallite               # Long context 7B
ollama pull stablelm-zephyr           # Lightweight chat 3B
ollama pull internvl2                 # Vision-language
ollama pull internvl3                 # Vision-language
ollama pull pixtral                   # Mistral vision

----

## Requirements

- macOS 20+
- Bash 3.2+

---

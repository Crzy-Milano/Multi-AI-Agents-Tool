#!/usr/bin/env python3
"""
Multi-Agent Orchestrator CLI
Laat meerdere LLM-modellen parallel samenwerken aan een taak.
Ondersteunt: Groq (gratis), Anthropic, of elke OpenAI-compatibele API.
"""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

try:
    from openai import OpenAI
except ImportError:
    print("Installeer eerst: pip install openai")
    sys.exit(1)

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key":  "GROQ_API_KEY",
        "default_agent_models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        "default_orch_model": "llama-3.3-70b-versatile",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "env_key":  "ANTHROPIC_API_KEY",
        "default_agent_models": [
            "claude-haiku-4-5-20251001",
            "claude-haiku-4-5-20251001",
            "claude-haiku-4-5-20251001",
        ],
        "default_orch_model": "claude-sonnet-4-5",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key":  "OPENAI_API_KEY",
        "default_agent_models": ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"],
        "default_orch_model": "gpt-4o",
    },
}

CHAT_AGENTS = [
    {"name": "Analist",  "system": "Je bent een analytische AI. Analyseer de vraag grondig en geef een helder, gestructureerd antwoord. Wees beknopt (max 4 zinnen)."},
    {"name": "Creatief", "system": "Je bent een creatieve AI. Geef een originele, verrassende invalshoek op de vraag. Wees bondig (max 4 zinnen)."},
    {"name": "Criticus", "system": "Je bent een kritische AI. Wijs op mogelijke valkuilen, fouten of nuances. Wees beknopt (max 4 zinnen)."},
]

CODE_AGENTS = [
    {"name": "Implementatie", "system": "Je bent een programmeur. Schrijf beknopte, werkende code voor de taak. Voeg alleen korte inline comments toe."},
    {"name": "Alternatief",   "system": "Je bent een programmeur. Schrijf een alternatieve, compacte implementatie voor de taak."},
    {"name": "Reviewer",      "system": "Je bent een code reviewer. Geef 3 concrete verbeterpunten over kwaliteit, leesbaarheid en edge cases."},
]

ORCHESTRATOR_SYSTEM_CHAT = "Je bent een orchestrator AI. Je ontvangt de antwoorden van drie parallelle agents. Combineer die tot één helder, volledig antwoord. Verwijder doublures en behoud het beste van elke agent."
ORCHESTRATOR_SYSTEM_CODE = "Je bent een orchestrator AI. Je ontvangt twee code-implementaties en één code review. Combineer de beste elementen, verwerk de reviewpunten, en geef de aanbevolen eindcode met een korte toelichting."

RESET   = "\033[0m"; BOLD = "\033[1m"; CYAN = "\033[96m"; GREEN = "\033[92m"
YELLOW  = "\033[93m"; MAGENTA = "\033[95m"; GRAY = "\033[90m"
AGENT_COLORS = [CYAN, YELLOW, MAGENTA]

def color(text, *codes): return "".join(codes) + text + RESET
def header(title):
    line = "─" * (len(title) + 4)
    return f"\n{BOLD}┌{line}┐\n│  {title}  │\n└{line}┘{RESET}\n"

def call_model(client, model, system, user_msg, max_tokens=1024):
    r = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
    )
    return r.choices[0].message.content

def call_agent(client, agent, model, user_msg, index):
    try:    return index, agent["name"], call_model(client, model, agent["system"], user_msg)
    except Exception as e: return index, agent["name"], f"[FOUT: {e}]"

def run_agents_parallel(client, agents, models, user_msg):
    results = [None] * len(agents)
    def task(i):
        return call_agent(client, agents[i], models[i] if i < len(models) else models[-1], user_msg, i)
    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        for fut in [pool.submit(task, i) for i in range(len(agents))]:
            idx, name, result = fut.result(); results[idx] = (name, result)
    return results

def print_agent_result(index, name, model, result):
    col = AGENT_COLORS[index % len(AGENT_COLORS)]
    print(color(f"\n▶ Agent {index + 1} — {name}", BOLD, col))
    print(color(f"  model: {model}", GRAY)); print()
    if "```" in result:
        parts = result.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part.strip(): print(part.strip())
            else:
                lang_end = part.find("\n")
                code = part[lang_end+1:].strip() if lang_end != -1 else part.strip()
                print(color("  " + "\n  ".join(code.splitlines()), GRAY))
    else:
        print(result.strip())

def print_orchestrator(result, model):
    print(header("Orchestrator — eindresultaat"))
    print(color(f"model: {model}", GRAY)); print(); print(result.strip()); print()

def interactive_loop(client, mode, agent_models, orch_model, provider):
    agents = CHAT_AGENTS if mode == "chat" else CODE_AGENTS
    orch_system = ORCHESTRATOR_SYSTEM_CHAT if mode == "chat" else ORCHESTRATOR_SYSTEM_CODE
    print(header(f"Multi-Agent Orchestrator  [{provider.upper()} · {mode.upper()}]"))
    print(color("Agents:", BOLD))
    for i, a in enumerate(agents):
        m = agent_models[i] if i < len(agent_models) else agent_models[-1]
        print(f"  {AGENT_COLORS[i % len(AGENT_COLORS)]}{a['name']}{RESET}  →  {color(m, GRAY)}")
    print(color(f"\nOrchestrator  →  {orch_model}", BOLD))
    print(color("\nTik 'exit' om te stoppen.\n", GRAY))
    while True:
        try:    prompt = input(color("Jij  ▶  ", BOLD, GREEN)).strip()
        except (EOFError, KeyboardInterrupt): print("\nTot ziens!"); break
        if not prompt: continue
        if prompt.lower() in ("exit", "quit", "stop"): print("Tot ziens!"); break
        print(color("\nAgents aan het werk...", GRAY))
        results = run_agents_parallel(client, agents, agent_models, prompt)
        for i, (name, result) in enumerate(results):
            print_agent_result(i, name, agent_models[i] if i < len(agent_models) else agent_models[-1], result)
        print(color("\nOrchestrator combineert...", GRAY))
        orch_prompt = f'Vraag: "{prompt}"\n\n' + "\n\n".join(f"Agent {i+1} ({n}):\n{r}" for i,(n,r) in enumerate(results))
        try:    orch_result = call_model(client, orch_model, orch_system, orch_prompt)
        except Exception as e: orch_result = f"[ORCHESTRATOR FOUT: {e}]"
        print_orchestrator(orch_result, orch_model)

def run_once(client, prompt, mode, agent_models, orch_model, output_json=False):
    agents = CHAT_AGENTS if mode == "chat" else CODE_AGENTS
    orch_system = ORCHESTRATOR_SYSTEM_CHAT if mode == "chat" else ORCHESTRATOR_SYSTEM_CODE
    results = run_agents_parallel(client, agents, agent_models, prompt)
    orch_prompt = f'Vraag: "{prompt}"\n\n' + "\n\n".join(f"Agent {i+1} ({n}):\n{r}" for i,(n,r) in enumerate(results))
    try:    orch_result = call_model(client, orch_model, orch_system, orch_prompt)
    except Exception as e: orch_result = f"[ORCHESTRATOR FOUT: {e}]"
    if output_json:
        print(json.dumps({"prompt": prompt, "mode": mode,
            "agents": [{"name": n, "model": agent_models[i] if i < len(agent_models) else agent_models[-1], "result": r} for i,(n,r) in enumerate(results)],
            "orchestrator": {"model": orch_model, "result": orch_result}}, ensure_ascii=False, indent=2))
    else:
        for i, (name, result) in enumerate(results):
            print_agent_result(i, name, agent_models[i] if i < len(agent_models) else agent_models[-1], result)
        print_orchestrator(orch_result, orch_model)

def main():
    p = argparse.ArgumentParser(description="Multi-Agent Orchestrator", formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Voorbeelden:
  python multi_agent.py --provider groq
  python multi_agent.py --provider groq --mode code --prompt "Schrijf een linked list in Python"
  python multi_agent.py --provider groq --agent-models llama-3.3-70b-versatile llama-3.1-8b-instant mixtral-8x7b-32768

Gratis Groq-modellen:
  llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it
""")
    p.add_argument("--provider", choices=list(PROVIDERS.keys()), default="groq")
    p.add_argument("--mode", choices=["chat", "code"], default="chat")
    p.add_argument("--agent-models", nargs="+", default=None, metavar="MODEL")
    p.add_argument("--orch-model", default=None, metavar="MODEL")
    p.add_argument("--prompt", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--api-key", default=None)
    args = p.parse_args()

    cfg = PROVIDERS[args.provider]
    api_key = args.api_key or os.environ.get(cfg["env_key"])
    if not api_key:
        print(color(f"Fout: geen API-sleutel gevonden voor {args.provider}.", BOLD))
        if args.provider == "groq":
            print("Maak een gratis account aan op https://console.groq.com")
            print(f"Stel daarna in: export {cfg['env_key']}=\"gsk_...\"")
        else:
            print(f"Stel in: export {cfg['env_key']}=\"<sleutel>\"")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    agent_models = args.agent_models or cfg["default_agent_models"]
    orch_model   = args.orch_model   or cfg["default_orch_model"]

    if args.prompt:
        run_once(client, args.prompt, args.mode, agent_models, orch_model, args.json)
    else:
        interactive_loop(client, args.mode, agent_models, orch_model, args.provider)

if __name__ == "__main__":
    main()

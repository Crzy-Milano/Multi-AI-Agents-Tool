#!/usr/bin/env python3
"""
Multi-Agent Orchestrator CLI
Meerdere AI-providers (Groq, OpenAI, Gemini) samenwerken in één sessie.
Live model-switching met !model commando's.
"""

import os, sys, json, argparse
from concurrent.futures import ThreadPoolExecutor

try:
    from openai import OpenAI
except ImportError:
    print("Installeer eerst: pip install openai google-generativeai")
    sys.exit(1)

# ─── Providers ────────────────────────────────────────────────────────────────

PROVIDER_CONFIGS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key":  "GROQ_API_KEY",
        "models":   ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "protocol": "openai",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key":  "OPENAI_API_KEY",
        "models":   ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "protocol": "openai",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key":  "GEMINI_API_KEY",
        "models":   ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "protocol": "openai",
    },
}

# Standaard opstelling: één agent per provider
DEFAULT_AGENTS = [
    {"provider": "groq",   "model": "llama-3.3-70b-versatile"},
    {"provider": "openai", "model": "gpt-4o-mini"},
    {"provider": "gemini", "model": "gemini-2.0-flash"},
]
DEFAULT_ORCH = {"provider": "groq", "model": "llama-3.3-70b-versatile"}

# ─── Agent-rollen ─────────────────────────────────────────────────────────────

CHAT_ROLES = [
    {"name": "Analist",  "system": "Je bent een analytische AI. Analyseer de vraag grondig en geef een helder, gestructureerd antwoord. Wees beknopt (max 4 zinnen)."},
    {"name": "Creatief", "system": "Je bent een creatieve AI. Geef een originele, verrassende invalshoek op de vraag. Wees bondig (max 4 zinnen)."},
    {"name": "Criticus", "system": "Je bent een kritische AI. Wijs op mogelijke valkuilen, fouten of nuances. Wees beknopt (max 4 zinnen)."},
]
CODE_ROLES = [
    {"name": "Implementatie", "system": "Je bent een programmeur. Schrijf beknopte, werkende code. Voeg alleen korte inline comments toe."},
    {"name": "Alternatief",   "system": "Je bent een programmeur. Schrijf een alternatieve, compacte implementatie."},
    {"name": "Reviewer",      "system": "Je bent een code reviewer. Geef 3 concrete verbeterpunten over kwaliteit, leesbaarheid en edge cases."},
]
ORCH_SYSTEM_CHAT = "Je bent een orchestrator AI. Combineer de antwoorden van de agents tot één helder, volledig antwoord. Verwijder doublures, behoud het beste van elke agent."
ORCH_SYSTEM_CODE = "Je bent een orchestrator AI. Combineer de twee code-implementaties en de review tot één aanbevolen eindcode met korte toelichting."

# ─── Kleuren ──────────────────────────────────────────────────────────────────

R="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"; GREEN="\033[92m"
YELLOW="\033[93m"; MAGENTA="\033[95m"; GRAY="\033[90m"; RED="\033[91m"
AGENT_COLORS = [CYAN, YELLOW, MAGENTA]

PROVIDER_COLORS = {"groq": CYAN, "openai": GREEN, "gemini": YELLOW}

def c(text, *codes): return "".join(codes) + text + R
def header(title):
    line = "─" * (len(title) + 4)
    return f"\n{BOLD}┌{line}┐\n│  {title}  │\n└{line}┘{R}\n"

# ─── Client cache ─────────────────────────────────────────────────────────────

_clients = {}

def get_client(provider, api_key=None):
    if provider in _clients:
        return _clients[provider]
    cfg = PROVIDER_CONFIGS[provider]
    key = api_key or os.environ.get(cfg["env_key"])
    if not key:
        return None
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    _clients[provider] = client
    return client

# ─── API-aanroep ──────────────────────────────────────────────────────────────

def call_model(provider, model, system, user_msg, max_tokens=1024):
    client = get_client(provider)
    if not client:
        raise Exception(f"Geen API-sleutel voor {provider} (stel ${PROVIDER_CONFIGS[provider]['env_key']} in)")
    r = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
    )
    return r.choices[0].message.content

def call_agent_task(role, agent_cfg, user_msg, index):
    try:
        result = call_model(agent_cfg["provider"], agent_cfg["model"], role["system"], user_msg)
        return index, role["name"], result, None
    except Exception as e:
        return index, role["name"], None, str(e)

def run_parallel(roles, agent_cfgs, user_msg):
    results = [None] * len(roles)
    def task(i): return call_agent_task(roles[i], agent_cfgs[i], user_msg, i)
    with ThreadPoolExecutor(max_workers=len(roles)) as pool:
        for fut in [pool.submit(task, i) for i in range(len(roles))]:
            idx, name, result, err = fut.result()
            results[idx] = (name, result, err)
    return results

# ─── Weergave ─────────────────────────────────────────────────────────────────

def print_agent(index, name, agent_cfg, result, err):
    col = AGENT_COLORS[index % len(AGENT_COLORS)]
    pcol = PROVIDER_COLORS.get(agent_cfg["provider"], GRAY)
    print(c(f"\n▶ Agent {index+1} — {name}", BOLD, col), end="")
    print(c(f"  [{agent_cfg['provider']} / {agent_cfg['model']}]", pcol))
    print()
    if err:
        print(c(f"  Fout: {err}", RED))
        return
    text = result or ""
    if "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part.strip(): print(part.strip())
            else:
                lang_end = part.find("\n")
                code = part[lang_end+1:].strip() if lang_end != -1 else part.strip()
                print(c("  " + "\n  ".join(code.splitlines()), GRAY))
    else:
        print(text.strip())

def print_orch(result, orch_cfg):
    pcol = PROVIDER_COLORS.get(orch_cfg["provider"], GRAY)
    print(header("Orchestrator — eindresultaat"))
    print(c(f"[{orch_cfg['provider']} / {orch_cfg['model']}]", pcol))
    print()
    print(result.strip())
    print()

def print_status(agent_cfgs, orch_cfg, mode):
    print(c("\nHuidige configuratie:", BOLD))
    print(f"  Modus: {mode}")
    roles = CHAT_ROLES if mode == "chat" else CODE_ROLES
    for i, (role, cfg) in enumerate(zip(roles, agent_cfgs)):
        pcol = PROVIDER_COLORS.get(cfg["provider"], GRAY)
        print(f"  Agent {i+1} ({role['name']}): {c(cfg['provider'], pcol)} / {cfg['model']}")
    pcol = PROVIDER_COLORS.get(orch_cfg["provider"], GRAY)
    print(f"  Orchestrator: {c(orch_cfg['provider'], pcol)} / {orch_cfg['model']}")
    print()

def print_help():
    print(c("\nCommando's:", BOLD))
    print("  !status                        — toon huidige model-configuratie")
    print("  !model <n> <provider> <model>  — verander model van agent n (1/2/3)")
    print("  !orch <provider> <model>       — verander orchestrator-model")
    print("  !mode chat|code                — wissel tussen chat en code-modus")
    print("  !models                        — toon alle beschikbare modellen")
    print("  !help                          — toon dit overzicht")
    print("  exit                           — stoppen")
    print()
    print(c("Voorbeeld:", GRAY))
    print(c("  !model 1 gemini gemini-1.5-pro", GRAY))
    print(c("  !orch openai gpt-4o", GRAY))
    print()

def print_models():
    print(c("\nBeschikbare modellen per provider:", BOLD))
    for name, cfg in PROVIDER_CONFIGS.items():
        pcol = PROVIDER_COLORS.get(name, GRAY)
        key_set = bool(os.environ.get(cfg["env_key"]))
        status = c("✓ sleutel gevonden", GREEN) if key_set else c("✗ geen sleutel", RED)
        print(f"\n  {c(name.upper(), BOLD, pcol)}  ({status})")
        for m in cfg["models"]:
            print(f"    {m}")
    print()

# ─── Commando-verwerking ──────────────────────────────────────────────────────

def handle_command(cmd, agent_cfgs, orch_cfg, mode):
    parts = cmd.strip().split()
    keyword = parts[0].lower()

    if keyword == "!status":
        print_status(agent_cfgs, orch_cfg, mode)

    elif keyword == "!help":
        print_help()

    elif keyword == "!models":
        print_models()

    elif keyword == "!mode":
        if len(parts) < 2 or parts[1] not in ("chat", "code"):
            print(c("Gebruik: !mode chat|code", RED)); return agent_cfgs, orch_cfg, mode
        mode = parts[1]
        print(c(f"Modus gewijzigd naar: {mode}", GREEN))

    elif keyword == "!model":
        if len(parts) < 4:
            print(c("Gebruik: !model <1|2|3> <provider> <model>", RED)); return agent_cfgs, orch_cfg, mode
        try:
            idx = int(parts[1]) - 1
            assert 0 <= idx <= 2
        except:
            print(c("Agent nummer moet 1, 2 of 3 zijn.", RED)); return agent_cfgs, orch_cfg, mode
        provider = parts[2].lower()
        model = parts[3]
        if provider not in PROVIDER_CONFIGS:
            print(c(f"Onbekende provider: {provider}. Kies uit: {', '.join(PROVIDER_CONFIGS)}", RED))
            return agent_cfgs, orch_cfg, mode
        client = get_client(provider)
        if not client:
            print(c(f"Geen API-sleutel voor {provider}. Stel ${PROVIDER_CONFIGS[provider]['env_key']} in.", RED))
            return agent_cfgs, orch_cfg, mode
        agent_cfgs[idx] = {"provider": provider, "model": model}
        pcol = PROVIDER_COLORS.get(provider, GRAY)
        print(c(f"Agent {idx+1} → {provider} / {model}", GREEN))

    elif keyword == "!orch":
        if len(parts) < 3:
            print(c("Gebruik: !orch <provider> <model>", RED)); return agent_cfgs, orch_cfg, mode
        provider = parts[1].lower()
        model = parts[2]
        if provider not in PROVIDER_CONFIGS:
            print(c(f"Onbekende provider: {provider}.", RED)); return agent_cfgs, orch_cfg, mode
        client = get_client(provider)
        if not client:
            print(c(f"Geen API-sleutel voor {provider}.", RED)); return agent_cfgs, orch_cfg, mode
        orch_cfg = {"provider": provider, "model": model}
        print(c(f"Orchestrator → {provider} / {model}", GREEN))

    else:
        print(c(f"Onbekend commando. Tik !help voor een overzicht.", RED))

    return agent_cfgs, orch_cfg, mode

# ─── Interactieve modus ───────────────────────────────────────────────────────

def interactive_loop(agent_cfgs, orch_cfg, mode):
    print(header("Multi-Agent Orchestrator"))
    print_status(agent_cfgs, orch_cfg, mode)
    print(c("Tik !help voor commando's, exit om te stoppen.\n", GRAY))

    while True:
        try:
            prompt = input(c("Jij  ▶  ", BOLD, GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTot ziens!"); break

        if not prompt: continue
        if prompt.lower() in ("exit", "quit", "stop"): print("Tot ziens!"); break
        if prompt.startswith("!"):
            agent_cfgs, orch_cfg, mode = handle_command(prompt, agent_cfgs, orch_cfg, mode)
            continue

        roles = CHAT_ROLES if mode == "chat" else CODE_ROLES
        orch_system = ORCH_SYSTEM_CHAT if mode == "chat" else ORCH_SYSTEM_CODE

        print(c("\nAgents aan het werk...", GRAY))
        results = run_parallel(roles, agent_cfgs, prompt)

        for i, (name, result, err) in enumerate(results):
            print_agent(i, name, agent_cfgs[i], result, err)

        valid = [(n, r) for n, r, e in results if r]
        if not valid:
            print(c("\nAlle agents gaven een fout — orchestrator overgeslagen.", RED)); continue

        print(c("\nOrchestrator combineert...", GRAY))
        orch_prompt = f'Vraag: "{prompt}"\n\n' + "\n\n".join(
            f"Agent {i+1} ({n}) [{agent_cfgs[i]['provider']}]:\n{r}"
            for i, (n, r, e) in enumerate(results) if r
        )
        try:
            orch_result = call_model(orch_cfg["provider"], orch_cfg["model"], orch_system, orch_prompt)
        except Exception as e:
            orch_result = f"[ORCHESTRATOR FOUT: {e}]"
        print_orch(orch_result, orch_cfg)

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Multi-Agent Orchestrator — Groq + OpenAI + Gemini samen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
API-sleutels instellen (gratis tiers beschikbaar):
  export GROQ_API_KEY="gsk_..."        # console.groq.com
  export OPENAI_API_KEY="sk-..."       # platform.openai.com
  export GEMINI_API_KEY="AIza..."      # aistudio.google.com

Voorbeelden:
  python3 multi_agent.py
  python3 multi_agent.py --mode code
  python3 multi_agent.py --agent groq llama-3.3-70b-versatile --agent openai gpt-4o-mini --agent gemini gemini-2.0-flash
  python3 multi_agent.py --orch openai gpt-4o

Live commando's tijdens sessie:
  !model 1 gemini gemini-1.5-pro
  !orch groq llama-3.3-70b-versatile
  !mode code
  !models
  !status
""")
    p.add_argument("--mode", choices=["chat","code"], default="chat")
    p.add_argument("--agent", nargs=2, action="append", metavar=("PROVIDER","MODEL"),
                   help="Agent instelling: --agent groq llama-3.3-70b-versatile (herhaalbaar)")
    p.add_argument("--orch", nargs=2, metavar=("PROVIDER","MODEL"), default=None)
    args = p.parse_args()

    # Bouw agent-configuratie
    if args.agent:
        agent_cfgs = [{"provider": a[0], "model": a[1]} for a in args.agent[:3]]
        while len(agent_cfgs) < 3:
            agent_cfgs.append(DEFAULT_AGENTS[len(agent_cfgs)])
    else:
        agent_cfgs = [dict(a) for a in DEFAULT_AGENTS]

    orch_cfg = {"provider": args.orch[0], "model": args.orch[1]} if args.orch else dict(DEFAULT_ORCH)

    # Controleer of er minstens één sleutel beschikbaar is
    any_key = any(os.environ.get(PROVIDER_CONFIGS[a["provider"]]["env_key"]) for a in agent_cfgs)
    if not any_key:
        print(c("Geen enkele API-sleutel gevonden!", BOLD))
        print("Stel minimaal één in:")
        print("  export GROQ_API_KEY=\"gsk_...\"    (gratis via console.groq.com)")
        print("  export GEMINI_API_KEY=\"AIza...\"  (gratis via aistudio.google.com)")
        print("  export OPENAI_API_KEY=\"sk-...\"   (betaald via platform.openai.com)")
        sys.exit(1)

    interactive_loop(agent_cfgs, orch_cfg, args.mode)

if __name__ == "__main__":
    main()

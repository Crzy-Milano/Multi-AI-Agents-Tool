#!/usr/bin/env python3
"""
Multi-Agent Orchestrator CLI — v4
Providers: Groq, OpenAI, Gemini, Mistral, Cohere, Together AI, Ollama (lokaal), LM Studio (lokaal)
"""

import os, sys, argparse
from concurrent.futures import ThreadPoolExecutor

try:
    from openai import OpenAI
except ImportError:
    print("Installeer eerst: pip install openai")
    sys.exit(1)

# ─── Providers ────────────────────────────────────────────────────────────────

PROVIDER_CONFIGS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key":  "GROQ_API_KEY",
        "free":     True,
        "signup":   "console.groq.com",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-4-scout",
            "qwen/qwen3-32b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_key":  "MISTRAL_API_KEY",
        "free":     True,
        "signup":   "console.mistral.ai",
        "models": [
            "mistral-small-latest",
            "mistral-medium-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b",
            "codestral-latest",
        ],
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key":  "TOGETHER_API_KEY",
        "free":     True,
        "signup":   "api.together.ai",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-Free",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            "google/gemma-2-27b-it",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        ],
    },
    "cohere": {
        "base_url": "https://api.cohere.com/compatibility/v1",
        "env_key":  "COHERE_API_KEY",
        "free":     True,
        "signup":   "dashboard.cohere.com",
        "models": [
            "command-r-plus",
            "command-r",
            "command-light",
            "command",
        ],
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key":  "GEMINI_API_KEY",
        "free":     True,
        "signup":   "aistudio.google.com",
        "models": [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.5-pro-preview-05-06",
        ],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key":  "OPENAI_API_KEY",
        "free":     False,
        "signup":   "platform.openai.com",
        "models":   ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env_key":  None,
        "free":     True,
        "local":    True,
        "signup":   "ollama.com",
        "models": [
            "llama3.3",
            "llama3.2",
            "llama3.1",
            "mistral",
            "gemma3",
            "gemma2",
            "qwen2.5",
            "deepseek-r1",
            "phi4",
            "codellama",
            "nomic-embed-text",
        ],
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "env_key":  None,
        "free":     True,
        "local":    True,
        "signup":   "lmstudio.ai",
        "models": [
            "local-model",
            "llama-3.2-3b-instruct",
            "mistral-7b-instruct",
            "phi-3-mini",
            "gemma-2-2b-it",
        ],
    },
}

DEFAULT_AGENTS = [
    {"provider": "groq",    "model": "llama-3.3-70b-versatile"},
    {"provider": "groq",    "model": "llama-3.1-8b-instant"},
    {"provider": "groq",    "model": "qwen/qwen3-32b"},
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
BLUE="\033[94m"; WHITE="\033[97m"
AGENT_COLORS = [CYAN, YELLOW, MAGENTA]
PROVIDER_COLORS = {
    "groq": CYAN, "openai": GREEN, "gemini": YELLOW,
    "mistral": MAGENTA, "together": BLUE, "cohere": "\033[38;5;208m",
    "ollama": "\033[38;5;46m", "lmstudio": "\033[38;5;51m",
}

def c(text, *codes): return "".join(codes) + text + R
def header(title):
    line = "─" * (len(title) + 4)
    return f"\n{BOLD}┌{line}┐\n│  {title}  │\n└{line}┘{R}\n"

# ─── Client cache ─────────────────────────────────────────────────────────────

_clients = {}

def get_client(provider):
    if provider in _clients:
        return _clients[provider]
    cfg = PROVIDER_CONFIGS[provider]
    local = cfg.get("local", False)

    if local:
        # Lokale providers hebben geen API-sleutel nodig
        client = OpenAI(api_key="local", base_url=cfg["base_url"])
        _clients[provider] = client
        return client

    key = os.environ.get(cfg["env_key"]) if cfg["env_key"] else None
    if not key:
        return None
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    _clients[provider] = client
    return client

def provider_available(provider):
    cfg = PROVIDER_CONFIGS[provider]
    if cfg.get("local", False):
        return True  # altijd beschikbaar als Ollama/LM Studio draait
    return get_client(provider) is not None

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
        print(c(f"  Fout: {err}", RED)); return
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
    print(); print(result.strip()); print()

def print_status(agent_cfgs, orch_cfg, mode):
    roles = CHAT_ROLES if mode == "chat" else CODE_ROLES
    print(c("\n┌─ Huidige configuratie ─────────────────────", BOLD))
    print(f"│  Modus: {c(mode, BOLD)}")
    for i, (role, cfg) in enumerate(zip(roles, agent_cfgs)):
        pcol = PROVIDER_COLORS.get(cfg["provider"], GRAY)
        print(f"│  Agent {i+1} ({role['name']}): {c(cfg['provider'], pcol)} / {cfg['model']}")
    pcol = PROVIDER_COLORS.get(orch_cfg["provider"], GRAY)
    print(f"│  Orchestrator: {c(orch_cfg['provider'], pcol)} / {orch_cfg['model']}")
    print(c("└────────────────────────────────────────────\n", BOLD))

def print_help():
    print(c("\n┌─ Commando's ───────────────────────────────────────────────────", BOLD))
    print("│  -models / !models              — interactief model-keuzemenu")
    print("│  !status                        — toon huidige configuratie")
    print("│  !model <n> <provider> <model>  — verander agent n direct")
    print("│  !orch <provider> <model>       — verander orchestrator direct")
    print("│  !mode chat|code                — wissel modus")
    print("│  !key <provider> <sleutel>      — voeg API-sleutel toe")
    print("│  !providers                     — toon alle providers & status")
    print("│  !help                          — toon dit overzicht")
    print("│  exit                           — stoppen")
    print(c("└────────────────────────────────────────────────────────────────\n", BOLD))

def print_providers():
    print(c("\n┌─ Beschikbare providers ─────────────────────────────────────────", BOLD))
    for name, cfg in PROVIDER_CONFIGS.items():
        pcol = PROVIDER_COLORS.get(name, GRAY)
        local = cfg.get("local", False)
        free = cfg.get("free", False)
        available = provider_available(name)

        if local:
            status = c("⌂ lokaal", BLUE)
        elif available:
            status = c("✓ verbonden", GREEN)
        else:
            status = c("✗ geen sleutel", RED)

        tags = []
        if free: tags.append(c("gratis", GREEN))
        if local: tags.append(c("offline", BLUE))
        tag_str = "  " + "  ".join(tags) if tags else ""

        print(f"│  {c(name.ljust(12), BOLD, pcol)}  {status}{tag_str}")
        if not local and not available and cfg["env_key"]:
            print(f"│  {'':12}  → export {cfg['env_key']}=\"...\"  ({cfg['signup']})")
    print(c("└────────────────────────────────────────────────────────────────\n", BOLD))

# ─── Interactief model-menu ───────────────────────────────────────────────────

def models_menu(agent_cfgs, orch_cfg, mode):
    roles = CHAT_ROLES if mode == "chat" else CODE_ROLES

    while True:
        print(header("Model-keuzemenu"))

        for i, (role, cfg) in enumerate(zip(roles, agent_cfgs)):
            pcol = PROVIDER_COLORS.get(cfg["provider"], GRAY)
            print(f"  {c(str(i+1), BOLD)}  Agent {i+1} — {role['name']}")
            print(f"     Provider : {c(cfg['provider'], pcol)}")
            print(f"     Model    : {cfg['model']}")
            print()

        pcol = PROVIDER_COLORS.get(orch_cfg["provider"], GRAY)
        print(f"  {c('O', BOLD)}  Orchestrator")
        print(f"     Provider : {c(orch_cfg['provider'], pcol)}")
        print(f"     Model    : {orch_cfg['model']}")
        print()

        print(c("  Kies agent om te wijzigen (1/2/3/O) of Enter om terug te gaan: ", GRAY), end="")
        try:
            choice = input().strip().upper()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "": print(c("Terug naar de chat.\n", GRAY)); break
        if choice not in ("1", "2", "3", "O"): print(c("Ongeldige keuze.\n", RED)); continue

        # Provider kiezen
        print()
        print(c("  Kies een provider:", BOLD))
        provider_list = list(PROVIDER_CONFIGS.keys())
        for j, name in enumerate(provider_list):
            cfg = PROVIDER_CONFIGS[name]
            pcol = PROVIDER_COLORS.get(name, GRAY)
            local = cfg.get("local", False)
            free = cfg.get("free", False)
            available = provider_available(name)

            if local:
                status = c("⌂ lokaal", BLUE)
            elif available:
                status = c("✓", GREEN)
            else:
                status = c("✗", RED)

            tags = []
            if free: tags.append(c("gratis", GREEN))
            if local: tags.append(c("offline", BLUE))
            tag_str = "  " + " ".join(tags) if tags else ""

            print(f"    {c(str(j+1), BOLD)}  {c(name.ljust(10), pcol)}  {status}{tag_str}")

        print()
        print(c("  Providernummer (of Enter om te annuleren): ", GRAY), end="")
        try:
            p_choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            continue

        if not p_choice: continue
        try:
            p_idx = int(p_choice) - 1
            assert 0 <= p_idx < len(provider_list)
        except:
            print(c("Ongeldige keuze.\n", RED)); continue

        provider = provider_list[p_idx]
        cfg = PROVIDER_CONFIGS[provider]

        # API-sleutel vragen indien nodig
        if not cfg.get("local", False) and not provider_available(provider):
            print()
            print(c(f"  Geen sleutel voor {provider}.", RED))
            print(f"  Maak gratis account aan op {c(cfg['signup'], CYAN)}")
            print(f"  Voer je API-sleutel in (of Enter om te annuleren): ", end="")
            try:
                new_key = input().strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not new_key: print(c("Geannuleerd.\n", GRAY)); continue
            os.environ[cfg["env_key"]] = new_key
            _clients.pop(provider, None)
            if not provider_available(provider):
                print(c("Sleutel lijkt ongeldig.\n", RED)); continue
            print(c(f"  Sleutel voor {provider} opgeslagen!\n", GREEN))

        # Model kiezen
        models = cfg["models"]
        print()
        print(c(f"  Modellen voor {provider}:", BOLD))
        for j, m in enumerate(models):
            print(f"    {c(str(j+1), BOLD)}  {m}")
        print(f"    {c('A', BOLD)}  Eigen modelnaam invoeren")
        print()
        print(c("  Modelnummer (of Enter om te annuleren): ", GRAY), end="")
        try:
            m_choice = input().strip().upper()
        except (EOFError, KeyboardInterrupt):
            continue

        if not m_choice: continue
        if m_choice == "A":
            print(c("  Modelnaam: ", GRAY), end="")
            try:
                model = input().strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not model: continue
        else:
            try:
                m_idx = int(m_choice) - 1
                assert 0 <= m_idx < len(models)
                model = models[m_idx]
            except:
                print(c("Ongeldige keuze.\n", RED)); continue

        pcol = PROVIDER_COLORS.get(provider, GRAY)
        if choice == "O":
            orch_cfg = {"provider": provider, "model": model}
            print(c(f"\n  ✓ Orchestrator → {provider} / {model}\n", GREEN))
        else:
            idx = int(choice) - 1
            agent_cfgs[idx] = {"provider": provider, "model": model}
            print(c(f"\n  ✓ Agent {choice} → {provider} / {model}\n", GREEN))

    return agent_cfgs, orch_cfg

# ─── Commando-verwerking ──────────────────────────────────────────────────────

def handle_command(cmd, agent_cfgs, orch_cfg, mode):
    parts = cmd.strip().split()
    keyword = parts[0].lower()

    if keyword in ("-models", "!models"):
        agent_cfgs, orch_cfg = models_menu(agent_cfgs, orch_cfg, mode)
    elif keyword in ("!providers", "-providers"):
        print_providers()
    elif keyword == "!status":
        print_status(agent_cfgs, orch_cfg, mode)
    elif keyword == "!help":
        print_help()
    elif keyword == "!mode":
        if len(parts) < 2 or parts[1] not in ("chat", "code"):
            print(c("Gebruik: !mode chat|code", RED))
        else:
            mode = parts[1]
            print(c(f"Modus → {mode}", GREEN))
    elif keyword == "!key":
        if len(parts) < 3:
            print(c("Gebruik: !key <provider> <api-sleutel>", RED))
        else:
            provider = parts[1].lower()
            key = parts[2]
            if provider not in PROVIDER_CONFIGS:
                print(c(f"Onbekende provider: {provider}", RED))
            else:
                os.environ[PROVIDER_CONFIGS[provider]["env_key"]] = key
                _clients.pop(provider, None)
                if provider_available(provider):
                    print(c(f"Sleutel voor {provider} opgeslagen!", GREEN))
                else:
                    print(c("Sleutel lijkt ongeldig.", RED))
    elif keyword == "!model":
        if len(parts) < 4:
            print(c("Gebruik: !model <1|2|3> <provider> <model>", RED))
        else:
            try:
                idx = int(parts[1]) - 1; assert 0 <= idx <= 2
            except:
                print(c("Agent nummer moet 1, 2 of 3 zijn.", RED))
                return agent_cfgs, orch_cfg, mode
            provider, model = parts[2].lower(), parts[3]
            if provider not in PROVIDER_CONFIGS:
                print(c(f"Onbekende provider: {provider}", RED))
            elif not provider_available(provider):
                print(c(f"Geen sleutel voor {provider}. Gebruik: !key {provider} <sleutel>", RED))
            else:
                agent_cfgs[idx] = {"provider": provider, "model": model}
                print(c(f"Agent {idx+1} → {provider} / {model}", GREEN))
    elif keyword == "!orch":
        if len(parts) < 3:
            print(c("Gebruik: !orch <provider> <model>", RED))
        else:
            provider, model = parts[1].lower(), parts[2]
            if provider not in PROVIDER_CONFIGS:
                print(c(f"Onbekende provider: {provider}", RED))
            elif not provider_available(provider):
                print(c(f"Geen sleutel voor {provider}.", RED))
            else:
                orch_cfg = {"provider": provider, "model": model}
                print(c(f"Orchestrator → {provider} / {model}", GREEN))
    else:
        print(c("Onbekend commando. Tik !help voor een overzicht.", RED))

    return agent_cfgs, orch_cfg, mode

# ─── Interactieve modus ───────────────────────────────────────────────────────

def interactive_loop(agent_cfgs, orch_cfg, mode):
    print(header("Multi-Agent Orchestrator  v4"))
    print_status(agent_cfgs, orch_cfg, mode)
    print(c("Tik -models om van model te wisselen, !help voor alle commando's.\n", GRAY))

    while True:
        try:
            prompt = input(c("Jij  ▶  ", BOLD, GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTot ziens!"); break

        if not prompt: continue
        if prompt.lower() in ("exit", "quit", "stop"): print("Tot ziens!"); break
        if prompt.startswith("!") or prompt.startswith("-"):
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
        description="Multi-Agent Orchestrator v4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gratis API-sleutels:
  export GROQ_API_KEY="gsk_..."         console.groq.com
  export MISTRAL_API_KEY="..."          console.mistral.ai
  export TOGETHER_API_KEY="..."         api.together.ai
  export COHERE_API_KEY="..."           dashboard.cohere.com
  export GEMINI_API_KEY="AIza..."       aistudio.google.com

Lokaal (geen sleutel nodig):
  Ollama:    ollama.com  →  ollama run llama3.3
  LM Studio: lmstudio.ai

Starten:
  python3 multi_agent-4.py
  python3 multi_agent-4.py --mode code
  python3 multi_agent-4.py --agent groq llama-3.3-70b-versatile --agent mistral mistral-small-latest --agent together meta-llama/Llama-3.3-70B-Instruct-Turbo-Free

Tijdens de sessie:
  -models       — interactief model-keuzemenu
  !providers    — toon alle providers & status
  !status       — toon huidige configuratie
  !help         — alle commando's
""")
    p.add_argument("--mode", choices=["chat","code"], default="chat")
    p.add_argument("--agent", nargs=2, action="append", metavar=("PROVIDER","MODEL"))
    p.add_argument("--orch", nargs=2, metavar=("PROVIDER","MODEL"), default=None)
    args = p.parse_args()

    if args.agent:
        agent_cfgs = [{"provider": a[0], "model": a[1]} for a in args.agent[:3]]
        while len(agent_cfgs) < 3:
            agent_cfgs.append(dict(DEFAULT_AGENTS[len(agent_cfgs)]))
    else:
        agent_cfgs = [dict(a) for a in DEFAULT_AGENTS]

    orch_cfg = {"provider": args.orch[0], "model": args.orch[1]} if args.orch else dict(DEFAULT_ORCH)

    # Check of er minstens één sleutel is (lokale providers tellen altijd mee)
    any_key = any(
        PROVIDER_CONFIGS[a["provider"]].get("local", False) or
        os.environ.get(PROVIDER_CONFIGS[a["provider"]].get("env_key") or "")
        for a in agent_cfgs
    )
    if not any_key:
        print(c("Geen API-sleutel gevonden!", BOLD))
        print("Stel minimaal één in, bijv.:")
        print("  export GROQ_API_KEY=\"gsk_...\"    (gratis → console.groq.com)")
        print("  export GEMINI_API_KEY=\"AIza...\"  (gratis → aistudio.google.com)")
        print("\nOf draai lokaal met Ollama (geen sleutel nodig):")
        print("  python3 multi_agent-4.py --agent ollama llama3.3 --agent ollama mistral --agent ollama phi4")
        sys.exit(1)

    interactive_loop(agent_cfgs, orch_cfg, args.mode)

if __name__ == "__main__":
    main()

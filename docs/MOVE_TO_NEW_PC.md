# Moving MarketLab to another PC

You need about 20 minutes. Only the code lives on GitHub. Secrets and the tournament
database do not, so you copy those across yourself.

## 0. Stop the old machine first (on the laptop)

Two copies running at once would double API calls and split the tournament into two
diverging databases.

1. Double-click `STOP_TRADING.bat`. This engages the kill switch and stops the daemon
   cleanly.
2. Stop it starting again at login (PowerShell):
   ```
   schtasks /change /tn "MarketLab Tournament" /disable
   ```

## 1. Install the tools (on the new PC)

- **Git**: https://git-scm.com/download/win
- **uv** (the Python runner), in PowerShell:
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **Ollama** (local AI): https://ollama.com/download. Then pull the model the AI sleeves use:
  ```
  ollama pull qwen2.5:3b-instruct
  ```
  It is a small model, so on the GPU it is fast. The local arm is still weak
  (`FINDINGS.md`). With 8-12 GB of VRAM you can try a stronger one, e.g.
  `ollama pull qwen2.5:7b-instruct`, then set `ai.runtime_model` in
  `configs/default.yaml`. Don't use `gpt-oss:20b`: it needs ~13 GB and would not fit.

## 2. Get the code

```
git clone https://github.com/sahilbaligar0407/PolymarketSearcher.git Kalshi
cd Kalshi
git checkout main
```

## 3. Recreate the secrets (never committed)

Create `.env` in the repo root with these names, copying the values from the laptop's
`.env`:

```
KALSHI_API_KEY_ID=
KALSHI_PRIVATE_KEY_PATH=C:\path\to\Kalshi\Production.txt
OPENAI_API_KEY=
OPENAI_MODEL=
OPENAI_DAILY_BUDGET_USD=
TYPESAFE_API_KEY=
FRED_API_KEY=
```

Also copy the Kalshi private key file (`Production.txt`) from the laptop. Point
`KALSHI_PRIVATE_KEY_PATH` at its new location; the repo root works, since it's
git-ignored. Optional keys are listed in `.env.example`.

## 4. Bring the tournament data (recommended)

This keeps every sleeve's bankroll and positions, the approved Polymarket-Kalshi pairs
(including all of Jev's match verdicts, so nothing is re-asked), trader scores and
history.

After step 0, copy these from the laptop's `Kalshi\data\` folder into the same place on
the PC:

- `marketlab.db` (~5.5 GB), plus `marketlab.db-wal` and `marketlab.db-shm` if they exist.
  Copy all of them together.
- `parquet\` (optional, but it holds the raw market and holdings history; large)

Use a USB drive or a network share. Skip this step to start the tournament from scratch
instead; everything rebuilds itself, but match discovery and trader scoring start over.

## 5. Start it

Double-click `START_TRADING.bat`. It:

- installs the Python environment,
- starts Ollama,
- clears the kill switch from step 0,
- opens the dashboard at http://127.0.0.1:8765, and
- runs the daemon under a watchdog that restarts it after any crash or hang.

Give it 2-3 minutes to boot. The dashboard should show RUNNING, with healthy feeds.

## 6. Start automatically after a reboot or power cut

PowerShell, run as your normal user (adjust the path):

```
schtasks /create /tn "MarketLab Tournament" /sc onlogon /rl limited ^
  /tr "cmd.exe /c \"C:\path\to\Kalshi\START_TRADING.bat\""
```

Also set Windows **Power & sleep** to *Never* sleep while plugged in. A sleeping PC
collects nothing. It resumes fine on wake, but the gap is lost.

## Day-to-day

- **Stop cleanly:** `STOP_TRADING.bat` (kill switch on) or `uv run marketlab paper stop`.
  The daemon picks up the stop request within 30 s.
- **Start again:** `START_TRADING.bat`.
- **Update code:** stop, `git pull`, start.
- **Health:** the dashboard, or `data\logs\marketlab.jsonl`.
- **What happened and why:** `docs/FINDINGS.md`. Entries 49-59 cover this deployment's
  fixes. Results before 2026-10-05 ~04:25 UTC are void.

uv venv
uv pip install .

python registry_server.py > registry_server.log 2>&1 &
python worker_agent_server.py > worker_agent_server.log 2>&1 &
python orchestrator_agent.py


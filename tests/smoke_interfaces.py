"""Opt-in HTTP and raw MCP stdio smoke; no research/model calls."""

import json
from pathlib import Path
from queue import Queue, Empty
import socket
import subprocess
import sys
import tempfile
from threading import Thread
from time import monotonic, sleep
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def streamlit_smoke(log):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true",
         f"--server.port={port}", "--browser.gatherUsageStats=false"],
        cwd=ROOT, stdout=log, stderr=log,
    )
    try:
        deadline = monotonic() + 60
        while True:
            try:
                with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
                    health_status, health_body = response.status, response.read().decode()
                break
            except OSError:
                if process.poll() is not None or monotonic() > deadline:
                    raise AssertionError("Streamlit did not become healthy")
                sleep(0.2)
        with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            root_status = response.status
        assert root_status == health_status == 200 and health_body == "ok"
        return {"root_http": root_status, "health_http": health_status, "health_body": health_body}
    finally:
        process.terminate()
        process.wait(timeout=20)


def mcp_smoke(log):
    process = subprocess.Popen(
        [sys.executable, "server.py"], cwd=ROOT, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8", bufsize=1,
    )
    lines = Queue()

    def read_stdout():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = Thread(target=read_stdout, daemon=True)
    reader.start()
    messages = []

    def send(message):
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    def receive(request_id):
        deadline = monotonic() + 120
        while monotonic() < deadline:
            try:
                line = lines.get(timeout=max(0.1, deadline - monotonic()))
            except Empty:
                break
            if line is None:
                raise AssertionError("MCP exited before responding")
            message = json.loads(line)  # Reject any stdout text outside JSON-RPC.
            assert message.get("jsonrpc") == "2.0"
            messages.append(message)
            if message.get("id") == request_id:
                assert "error" not in message, message
                return message["result"]
        raise AssertionError("MCP response timed out")

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "phase5-smoke", "version": "1.0"},
        }})
        initialize = receive(1)
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        discovered = receive(2)
        names = [tool["name"] for tool in discovered["tools"]]
        assert names == ["crew_research"], names
        process.stdin.close()  # Clean client disconnect.
        process.wait(timeout=30)
        reader.join(timeout=5)
        assert not reader.is_alive()
        while not lines.empty():
            line = lines.get_nowait()
            if line is not None:
                message = json.loads(line)
                assert message.get("jsonrpc") == "2.0"
                messages.append(message)
        assert process.returncode == 0
        return {"initialized": True, "tools": names, "json_rpc_only": True,
                "messages": len(messages), "clean_disconnect": True, "exit_code": process.returncode,
                "protocol_version": initialize["protocolVersion"]}
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=20)


if __name__ == "__main__":
    with tempfile.TemporaryFile(mode="w+b") as log:
        try:
            print(json.dumps({"streamlit": streamlit_smoke(log), "mcp": mcp_smoke(log)}, indent=2))
        except Exception:
            log.seek(0)
            sys.stderr.write(log.read().decode("utf-8", errors="replace"))
            raise

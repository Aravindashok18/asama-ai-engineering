import json

import httpx
import pytest

from hwqa.assistant import Assistant
from hwqa.backends import BackendError, OllamaBackend, OpenAICompatBackend, ToolCall, Turn, detect_tools
from hwqa.cli import main
from hwqa.grounding import ungrounded_claims


# ------------------------------------------------------------- rule backend


def test_brief_example_answer(thinkpad):
    ans = Assistant(thinkpad).ask("Can I add more RAM, and how much?")
    assert ans.text.startswith("You have 16 GB in 1 of 2 slots. The board supports up to 64 GB, "
                               "so you can add a stick of up to 32 GB")
    assert ans.unverified == []
    assert [c["name"] for c in ans.tool_calls] == ["get_memory_info"]


def test_follow_up_questions_keep_context(thinkpad):
    a = Assistant(thinkpad)
    a.ask("How much RAM do I have?")
    gpu = a.ask("and what about the GPU?")
    assert "Barcelo" in gpu.text and [c["name"] for c in gpu.tool_calls] == ["get_gpu_info"]
    again = a.ask("which driver does it use?")  # no topic word: reuse the last topic
    assert "amdgpu" in again.text
    assert again.tool_calls[0]["cached"] is True


def test_nvme_question(thinkpad, legion):
    assert Assistant(thinkpad).ask("Is my SSD NVMe?").text.startswith("Yes, /dev/nvme0n1 is an NVMe SSD")
    assert "HDD" in Assistant(legion).ask("what disks do I have").text


def test_says_i_dont_know_with_reason(thinkpad_no_root, vm):
    t = Assistant(thinkpad_no_root).ask("Can I add more RAM?").text
    assert "couldn't determine" in t and "needs root" in t and "64" not in t
    t = Assistant(vm).ask("What GPU do I have?").text
    assert "lspci" in t and "pciutils" in t


def test_overview_and_help(thinkpad):
    ans = Assistant(thinkpad).ask("give me a summary of my laptop specs")
    assert {c["name"] for c in ans.tool_calls} >= {"get_cpu_info", "get_memory_info", "get_gpu_info"}
    assert "I can answer questions" in Assistant(thinkpad).ask("hello").text


def test_detect_tools():
    assert detect_tools("how much space is left on my disk") == ["get_disk_usage"]
    assert detect_tools("is my ssd nvme") == ["get_storage_info"]
    assert detect_tools("battery cycles") == ["get_battery_info"]


# ------------------------------------------------------ LLM loop (scripted)


class ScriptedLLM:
    """Plays back a fixed sequence of turns; records what it was sent."""

    name = "scripted"
    local = True

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def complete(self, messages, tools):
        self.seen.append((json.loads(json.dumps(messages)), bool(tools)))
        return self.turns.pop(0)


def test_llm_tool_loop_and_grounded_answer(thinkpad):
    llm = ScriptedLLM([Turn(tool_calls=[ToolCall("c1", "get_memory_info", {})]),
                       Turn(content="You have 16 GB in 1 of 2 slots; you can add up to 32 GB.")])
    ans = Assistant(thinkpad, llm).ask("Can I add RAM?")
    assert ans.unverified == []
    tool_msg = llm.seen[1][0][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["data"]["upgrade"]["max_per_slot_gb"] == 32


def test_hallucinated_spec_triggers_retry_then_is_fixed(thinkpad):
    llm = ScriptedLLM([Turn(tool_calls=[ToolCall("c1", "get_memory_info", {})]),
                       Turn(content="Your board supports up to 128 GB."),          # invented
                       Turn(content="Your board supports up to 64 GB.")])           # corrected
    ans = Assistant(thinkpad, llm).ask("max RAM?")
    assert ans.text == "Your board supports up to 64 GB." and ans.unverified == []
    retry_prompt = llm.seen[2][0][-1]["content"]
    assert "128 GB" in retry_prompt


def test_unfixable_hallucination_is_flagged_to_the_user(thinkpad):
    llm = ScriptedLLM([Turn(tool_calls=[ToolCall("c1", "get_cpu_info", {})]),
                       Turn(content="It boosts to 5.9 GHz."), Turn(content="It boosts to 5.9 GHz.")])
    ans = Assistant(thinkpad, llm).ask("how fast?")
    assert ans.unverified == ["5.9 GHz"] and "could not verify" in ans.text


def test_model_cannot_run_unknown_tools(thinkpad):
    llm = ScriptedLLM([Turn(tool_calls=[ToolCall("c1", "run_command", {"cmd": "cat /etc/shadow"})]),
                       Turn(content="I can't do that.")])
    ans = Assistant(thinkpad, llm).ask("read /etc/shadow")
    assert "unknown tool" in ans.tool_calls[0]["error"]
    assert "shadow" not in json.dumps(llm.seen[1][0][-1]["content"]).replace("run_command", "")


def test_runaway_tool_calls_are_capped(thinkpad):
    calls = [Turn(tool_calls=[ToolCall(f"c{i}", "get_cpu_info", {})]) for i in range(4)]
    llm = ScriptedLLM(calls + [Turn(content="8 cores.")])
    ans = Assistant(thinkpad, llm, max_rounds=4).ask("cpu?")
    assert ans.text == "8 cores." and len(ans.tool_calls) == 4
    assert llm.seen[-1][1] is False  # final call made without tools


def test_ungrounded_claims_tolerances():
    evidence = [{"data": {"total_bytes": 16 * 1024**3, "max_mhz": 4545, "health_percent": 87}}]
    assert ungrounded_claims("16 GB RAM, up to 4.5 GHz, health 87%", evidence) == []
    assert ungrounded_claims("17.2 GB", evidence) == []  # 16 GiB written in decimal GB is still grounded
    assert ungrounded_claims("18.5 GB RAM and 4 cores", evidence) == ["18.5 GB", "4 cores"]


# -------------------------------------------------------- real backends (mocked HTTP)


def test_ollama_backend_request_and_tool_call_parsing():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_gpu_info", "arguments": {}}}]}})

    b = OllamaBackend()
    b.client = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(handler))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "gpu?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
             "function": {"name": "get_cpu_info", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "name": "get_cpu_info", "content": "{}"}]
    turn = b.complete(msgs, [{"type": "function", "function": {"name": "get_gpu_info"}}])
    assert turn.tool_calls[0].name == "get_gpu_info"
    body = seen["body"]
    assert body["options"]["temperature"] == 0 and body["tools"]
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == {}
    assert body["messages"][3] == {"role": "tool", "content": "{}", "tool_name": "get_cpu_info"}


def test_text_json_tool_call_fallback():
    b = OllamaBackend()
    b.client = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"message": {"content": '{"name": "get_battery_info", "arguments": {}}'}})))
    assert b.complete([{"role": "user", "content": "x"}], [{"x": 1}]).tool_calls[0].name == "get_battery_info"


def test_remote_model_needs_explicit_opt_in():
    with pytest.raises(BackendError):
        OllamaBackend(base_url="http://gpu-box.example.com:11434")
    with pytest.raises(BackendError):
        OpenAICompatBackend(model="m", base_url="https://api.example.com/v1")
    assert OpenAICompatBackend(model="m", base_url="https://api.example.com/v1", allow_remote=True).local is False


def test_openai_compat_parses_string_arguments():
    b = OpenAICompatBackend(model="m")
    b.client = httpx.Client(base_url="http://localhost:8080/v1", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "x", "function": {"name": "get_disk_usage", "arguments": '{"mountpoint": "/"}'}}]}}]})))
    turn = b.complete([{"role": "user", "content": "space?"}], [{"x": 1}])
    assert turn.tool_calls[0].arguments == {"mountpoint": "/"}


# -------------------------------------------------------------------- CLI


def test_cli_ask_json(capsys):
    main(["ask", "--backend", "rules", "--sudo", "--fixtures", "fixtures/thinkpad-e14", "--json", "Is my SSD NVMe?"])
    out = json.loads(capsys.readouterr().out)
    assert out["answer"].startswith("Yes") and out["tool_calls"][0]["name"] == "get_storage_info"


def test_cli_tools_and_dump(capsys):
    main(["tools", "--fixtures", "fixtures/minimal-vm"])
    assert "dmidecode not installed" in capsys.readouterr().out
    main(["dump", "--fixtures", "fixtures/legion-5"])
    data = json.loads(capsys.readouterr().out)
    assert data["get_gpu_info"]["data"]["count"] == 2

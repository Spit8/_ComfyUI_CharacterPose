"""Text prompt → joint Euler angles via OpenAI-compatible chat API."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .presets import get_action_angles

# Controllable bones (FK applies angles at these nodes)
ALLOWED_JOINTS: frozenset[str] = frozenset(
    {
        "spine",
        "chest",
        "neck",
        "head",
        "r_shoulder",
        "r_elbow",
        "r_wrist",
        "l_shoulder",
        "l_elbow",
        "l_wrist",
        "r_hip",
        "r_knee",
        "r_ankle",
        "l_hip",
        "l_knee",
        "l_ankle",
    }
)

# Per-axis clamp (degrees) after parse — keeps poses from breaking
_JOINT_LIMITS: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "spine": ((-40, 40), (-45, 45), (-30, 30)),
    "chest": ((-30, 30), (-35, 35), (-25, 25)),
    "neck": ((-35, 35), (-40, 40), (-30, 30)),
    "head": ((-35, 35), (-45, 45), (-30, 30)),
    "r_shoulder": ((-130, 130), (-90, 90), (-90, 90)),
    "l_shoulder": ((-130, 130), (-90, 90), (-90, 90)),
    "r_elbow": ((0, 140), (-40, 40), (-40, 40)),
    "l_elbow": ((0, 140), (-40, 40), (-40, 40)),
    "r_wrist": ((-60, 60), (-50, 50), (-50, 50)),
    "l_wrist": ((-60, 60), (-50, 50), (-50, 50)),
    "r_hip": ((-100, 60), (-40, 40), (-40, 40)),
    "l_hip": ((-100, 60), (-40, 40), (-40, 40)),
    "r_knee": ((0, 140), (-20, 20), (-20, 20)),
    "l_knee": ((0, 140), (-20, 20), (-20, 20)),
    "r_ankle": ((-40, 40), (-25, 25), (-25, 25)),
    "l_ankle": ((-40, 40), (-25, 25), (-25, 25)),
}

_DEFAULT_LIMIT = ((-120, 120), (-90, 90), (-90, 90))

SYSTEM_PROMPT = """You convert a short pose description into Euler joint angles for a humanoid kinematic skeleton.

Conventions:
- Y-up, character faces +Z at rest.
- Arms hang down along -Y at rest (not T-pose).
- Angles are degrees (rx, ry, rz) applied in XYZ Euler order at each joint.
- Shoulder rx: arm forward (negative) / back (positive).
- Elbow rx: positive bends the forearm toward the upper arm (0 = straight).
- Hip rx: thigh forward (negative) / back (positive).
- Knee rx: positive bends the shin.
- For walk/run: arms stay mostly low with moderate swing; do NOT raise both arms overhead.
- For fight/guard: raise fists near face height.
- For cast/jump: raised arms are OK when implied.

Allowed joints only: spine, chest, neck, head, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle.

Reply with ONLY valid JSON (no markdown), shape:
{"joints": {"r_shoulder": [rx, ry, rz], "...": [rx, ry, rz]}, "notes": "one short phrase"}
Omit joints that stay at rest. Values are absolute angles from the neutral hang-down rest pose (not deltas).
"""


class PosePromptError(RuntimeError):
    """User-visible failure resolving a text pose prompt."""


def resolve_api_key(explicit: str | None = None) -> str:
    key = (explicit or "").strip()
    if key:
        return key
    for env in ("CHARACTERPOSE_LLM_API_KEY", "OPENAI_API_KEY"):
        v = (os.environ.get(env) or "").strip()
        if v:
            return v
    return ""


def _clamp_axis(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def clamp_joint_angles(
    joints: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    """Clamp known joints to safe ranges; drop unknown names."""
    out: dict[str, tuple[float, float, float]] = {}
    for name, triple in joints.items():
        key = name.strip().lower()
        if key not in ALLOWED_JOINTS:
            continue
        limits = _JOINT_LIMITS.get(key, _DEFAULT_LIMIT)
        rx, ry, rz = float(triple[0]), float(triple[1]), float(triple[2])
        out[key] = (
            _clamp_axis(rx, limits[0][0], limits[0][1]),
            _clamp_axis(ry, limits[1][0], limits[1][1]),
            _clamp_axis(rz, limits[2][0], limits[2][1]),
        )
    return out


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise PosePromptError("LLM returned an empty response.")
    # Strip ```json fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: first {...} block
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise PosePromptError(f"LLM response is not JSON: {raw[:200]!r}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise PosePromptError(f"Failed to parse LLM JSON: {e}") from e
    if not isinstance(data, dict):
        raise PosePromptError("LLM JSON root must be an object.")
    return data


def parse_joints_payload(data: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    """Extract joint triples from LLM JSON object."""
    joints_raw = data.get("joints")
    if joints_raw is None and any(k in ALLOWED_JOINTS for k in data):
        joints_raw = {k: v for k, v in data.items() if k in ALLOWED_JOINTS}
    if not isinstance(joints_raw, dict):
        raise PosePromptError('LLM JSON must contain a "joints" object.')
    parsed: dict[str, tuple[float, float, float]] = {}
    for name, val in joints_raw.items():
        if not isinstance(name, str):
            continue
        if isinstance(val, dict):
            try:
                triple = (float(val["rx"]), float(val["ry"]), float(val["rz"]))
            except (KeyError, TypeError, ValueError):
                continue
        elif isinstance(val, (list, tuple)) and len(val) >= 3:
            try:
                triple = (float(val[0]), float(val[1]), float(val[2]))
            except (TypeError, ValueError):
                continue
        else:
            continue
        parsed[name] = triple
    return clamp_joint_angles(parsed)


def parse_llm_angles_text(text: str) -> dict[str, tuple[float, float, float]]:
    """Parse + clamp angles from raw LLM text (no network)."""
    return parse_joints_payload(_extract_json_object(text))


def chat_completions_json(
    *,
    prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: float = 60.0,
) -> str:
    """POST /chat/completions and return assistant message content."""
    base = (base_url or "").rstrip("/")
    if not base:
        raise PosePromptError("llm_base_url is empty.")
    if not api_key:
        raise PosePromptError(
            "Missing LLM API key. Set llm_api_key on the node, or env "
            "CHARACTERPOSE_LLM_API_KEY / OPENAI_API_KEY."
        )
    if not (prompt or "").strip():
        raise PosePromptError("pose_prompt is empty (text mode requires a description).")

    url = f"{base}/chat/completions"
    body = {
        "model": model or "gpt-4o-mini",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ],
        "response_format": {"type": "json_object"},
    }

    def _post(payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        payload = _post(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        # Some local servers (older Ollama) reject response_format
        if e.code in (400, 422) and "response_format" in body:
            body_fallback = dict(body)
            body_fallback.pop("response_format", None)
            try:
                payload = _post(body_fallback)
            except urllib.error.HTTPError as e2:
                detail2 = e2.read().decode("utf-8", errors="replace")[:400]
                raise PosePromptError(f"LLM HTTP {e2.code}: {detail2}") from e2
            except urllib.error.URLError as e2:
                raise PosePromptError(f"LLM request failed: {e2}") from e2
        else:
            raise PosePromptError(f"LLM HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise PosePromptError(f"LLM request failed: {e}") from e
    except TimeoutError as e:
        raise PosePromptError(f"LLM request timed out after {timeout_s}s.") from e
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise PosePromptError(f"Unexpected LLM response shape: {payload!r}") from e
    if not isinstance(content, str):
        raise PosePromptError("LLM message content is not a string.")
    return content


def angles_from_text_prompt(
    prompt: str,
    *,
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    seed_action: str = "idle",
    timeout_s: float = 60.0,
) -> dict[str, tuple[float, float, float]]:
    """Call LLM, parse joints, replace onto seed preset angles (absolute overrides)."""
    key = resolve_api_key(api_key)
    content = chat_completions_json(
        prompt=prompt,
        base_url=base_url,
        api_key=key,
        model=model,
        timeout_s=timeout_s,
    )
    overrides = parse_llm_angles_text(content)
    if not overrides:
        raise PosePromptError("LLM returned no usable joint angles.")
    base = get_action_angles(seed_action)
    # Absolute overrides (replace), not additive merge
    out = dict(base)
    out.update(overrides)
    return out


def angles_from_parsed_joints(
    joints: dict[str, tuple[float, float, float]],
    *,
    seed_action: str = "idle",
) -> dict[str, tuple[float, float, float]]:
    """Apply clamped joint overrides onto a seed preset (for tests / offline)."""
    base = get_action_angles(seed_action)
    out = dict(base)
    out.update(clamp_joint_angles(joints))
    return out

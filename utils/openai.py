import json
import os
import re

try:
    import openai
except Exception:
    openai = None

def call_openai_for_stress(features: dict, model: str = "gpt-4o-mini", timeout: int = 10) -> dict:
    if openai is None:
        raise RuntimeError("openai package not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL", model)

    system = (
        "You estimate human stress level from physiological HRV features.\n"
        "Inputs include heart rate variability metrics, heart rate statistics, "
        "frequency-domain features, nonlinear HRV features, and demographics.\n\n"
        "Reason about autonomic nervous system balance and the relationships "
        "between variability, heart rate patterns, and signal stability.\n\n"
        "Perform reasoning internally but DO NOT output the reasoning.\n\n"
        "Carefully calculate the stress level, "
        "if metrics indicate high stress: feel free to use high values, "
        "and if they indicate low stress: feel free to use low values.\n\n"
        "Return ONLY valid JSON with exactly these fields:\n"
        "{\n"
        '  "stress_level_pct": float between 0 and 100,\n'
        '  "is_stressed": boolean\n'
        "}\n\n"
        "is_stressed must be true if stress_level_pct >= 50.\n"
        "No explanations, markdown, or extra text.\n"
        "It's very important to treat each prediction as an independent evaluation."
    )

    user_msg = (
        "Features JSON:\n" + json.dumps(features, separators=(",", ":")) + "\n\n"
        "Respond with valid JSON exactly as described. Example: {\"stress_level_pct\": 42.1, \"is_stressed\": false}"
    )

    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = openai.OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            max_tokens=40,
            temperature=0.5,
            top_p=0.9,
            timeout=timeout,
        )
    except Exception as e:
        raise ValueError(f"OpenAI-compatible request failed: {e}")

    text = ""
    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise ValueError(f"Unexpected OpenAI response format: {e}")

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("OpenAI response did not contain a JSON object")

    parsed = json.loads(m.group(0))

    if "stress_level_pct" not in parsed or "is_stressed" not in parsed:
        raise ValueError("OpenAI JSON missing required keys")

    return {"stress_level_pct": float(parsed["stress_level_pct"]), "is_stressed": bool(parsed["is_stressed"])}

"""Optional Qwen3.5-4B adapter, distilled from the organizer's local Qwen baseline.

The HTTP service and its demo tests do not import torch. Heavy dependencies are
loaded only when MODEL_MODE=qwen starts this adapter.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Answer the remote-sensing image question using only the requested answer format. "
    "Do not explain."
)
MAX_NEW_TOKENS = {
    "yes_no": 8,
    "integer": 8,
    "enum": 16,
    "short_text": 24,
    "single_choice": 8,
    "bbox": 32,
}


def prompt_for_request(request: dict[str, Any], bbox_mode: str) -> tuple[str, str]:
    """Translate evaluator constraints into the same instructions as the local baseline."""
    constraint = request["response_constraint"]
    kind = constraint["type"]
    prefix = ""
    if len(request["images"]) == 2:
        prefix = (
            "The first image is before/Image A. The second image is after/Image B. "
            "Keep this temporal order.\n\n"
        )
    if kind == "bbox":
        if bbox_mode == "normalized_1000":
            instruction = (
                "Answer only with one valid JSON array [xmin, ymin, xmax, ymax] using "
                "coordinates on a normalized 0-1000 grid, where [0, 0] is the top-left "
                "and [1000, 1000] is the bottom-right of the image."
            )
        elif bbox_mode == "pixel_xyxy":
            instruction = (
                "Answer only with one valid JSON array [xmin, ymin, xmax, ymax] using "
                f"pixel coordinates in the original {request['image_width']} x "
                f"{request['image_height']} image."
            )
        else:
            raise ValueError("QWEN_BBOX_COORDINATES must be normalized_1000 or pixel_xyxy")
        token_class = "bbox"
    elif kind == "single_choice":
        choices = request["choices"]
        choice_text = "\n".join(f"{letter}. {choices[letter]}" for letter in sorted(choices))
        instruction = f"Choices:\n{choice_text}\nAnswer only with one letter from A to E."
        token_class = "single_choice"
    elif kind == "integer":
        instruction = "Answer only with one non-negative integer."
        token_class = "integer"
    elif kind == "short_text":
        instruction = "Answer with only a concise noun phrase."
        token_class = "short_text"
    elif kind == "enum":
        values = constraint["values"]
        if constraint.get("question_form") == "yes_no" and values == ["Yes", "No"]:
            instruction = "Answer only Yes or No."
            token_class = "yes_no"
        else:
            instruction = "Answer only with one of these exact values: " + ", ".join(values) + "."
            token_class = "enum"
    else:
        raise ValueError(f"unsupported constraint type: {kind}")
    return prefix + request["question"] + "\n\n" + instruction, token_class


def parse_generated_answer(raw: str, request: dict[str, Any], bbox_mode: str) -> str | list[int | float]:
    """Convert generated text into the API answer type; never invent a fallback answer."""
    if not isinstance(raw, str):
        raise ValueError("model output must be text")
    final = raw.rsplit("</think>", 1)[-1].strip()
    if not final or "<think>" in final:
        raise ValueError("model returned an empty or unfinished thinking answer")
    constraint = request["response_constraint"]
    kind = constraint["type"]
    if kind == "bbox":
        answer = json.loads(final)
        if not (
            isinstance(answer, list) and len(answer) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in answer)
            and answer[0] < answer[2] and answer[1] < answer[3]
        ):
            raise ValueError("bbox must be four finite xyxy numbers")
        if bbox_mode == "normalized_1000":
            if any(not 0 <= v <= 1000 for v in answer):
                raise ValueError("normalized bbox must be in [0, 1000]")
            width, height = request["image_width"], request["image_height"]
            answer = [
                round(answer[0] * width / 1000),
                round(answer[1] * height / 1000),
                round(answer[2] * width / 1000),
                round(answer[3] * height / 1000),
            ]
        elif bbox_mode != "pixel_xyxy":
            raise ValueError("unsupported bbox coordinate mode")
        if not answer[0] < answer[2] or not answer[1] < answer[3]:
            raise ValueError("bbox collapsed after coordinate conversion")
        return answer
    if "\n" in final:
        raise ValueError("text answer must be a single line")
    if kind in {"enum", "single_choice"}:
        for value in constraint["values"]:
            if final.casefold() == value.casefold():
                return value
        raise ValueError("answer is not one of the allowed values")
    if kind == "integer":
        if re.fullmatch(r"(?:0|[1-9]\d*)", final) is None:
            raise ValueError("integer answer is not a non-negative integer")
        return final
    if kind == "short_text":
        if len(final) > 4096:
            raise ValueError("answer is too long")
        return final
    raise ValueError(f"unsupported constraint type: {kind}")


class QwenModelAdapter:
    """Load one local Qwen3.5-4B model and answer protocol-2.0 requests."""

    def __init__(self) -> None:
        model_dir_text = os.environ.get("QWEN_MODEL_DIR", "")
        if not model_dir_text:
            raise ValueError("set QWEN_MODEL_DIR to a local Qwen3.5-4B model directory")
        model_dir = Path(model_dir_text).expanduser()
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError("QWEN_MODEL_DIR must contain config.json")
        self.bbox_mode = os.environ.get("QWEN_BBOX_COORDINATES", "normalized_1000")
        if self.bbox_mode not in {"normalized_1000", "pixel_xyxy"}:
            raise ValueError("QWEN_BBOX_COORDINATES must be normalized_1000 or pixel_xyxy")

        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

        if not torch.cuda.is_available():
            raise RuntimeError("this Qwen example requires a CUDA GPU")
        set_seed(20260715, deterministic=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        self.Image = Image
        self.lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()

    def predict(self, request: dict[str, Any], image_paths: list[Path]) -> str | list[int | float]:
        if len(image_paths) != len(request["images"]):
            raise ValueError("image count does not match request")
        prompt, token_class = prompt_for_request(request, self.bbox_mode)
        images = []
        try:
            for path in image_paths:
                with self.Image.open(path) as source:
                    images.append(source.convert("RGB").copy())
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": prompt},
                ]},
            ]
            with self.lock:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )
                moved = {
                    key: value.to(
                        device="cuda:0",
                        **({"dtype": self.torch.bfloat16} if getattr(value, "is_floating_point", lambda: False)() else {}),
                    ) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                with self.torch.inference_mode():
                    output_ids = self.model.generate(
                        **moved,
                        do_sample=False,
                        max_new_tokens=MAX_NEW_TOKENS[token_class],
                        use_cache=True,
                    )
                prompt_length = int(moved["input_ids"].shape[-1])
                raw = self.processor.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
            return parse_generated_answer(raw, request, self.bbox_mode)
        finally:
            for image in images:
                image.close()

import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .playbook import ROUTES, messages

ROOT = Path(__file__).resolve().parents[1]
BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
LOCAL_SETTINGS = ROOT / "settings.local.json"
MODEL_DIR = Path(
    os.environ.get("WORKPATTERN_MODEL_DIR")
    or (json.loads(LOCAL_SETTINGS.read_text(encoding="utf-8-sig"))["model_dir"] if LOCAL_SETTINGS.is_file() else ROOT / "weights")
).expanduser()
LOCAL_BASE = MODEL_DIR / "qwen"
BUNDLED_ADAPTER = ROOT / "model/adapter"
MAX_LENGTH = 1024
MAX_FEEDBACK = 500


def load_base(device: str):
    local = (LOCAL_BASE / "model.safetensors").is_file()
    source = str(LOCAL_BASE) if local else BASE_ID
    options = {} if local else {"revision": REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, padding_side="left", **options)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        source, torch_dtype=dtype, use_safetensors=True, **options
    ).to(device)
    return model, tokenizer


def batch_inputs(tokenizer, feedback, device):
    prompts = [
        tokenizer.apply_chat_template(messages(item), tokenize=False, add_generation_prompt=True)
        for item in feedback
    ]
    inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
    if inputs.input_ids.shape[-1] > MAX_LENGTH:
        raise ValueError("反馈超出模型输入长度，请缩短后重试。")
    return inputs.to(device)


def token_ids(tokenizer) -> list[int]:
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in ROUTES]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("Each route label must use one token.")
    return [ids[0] for ids in labels]


def next_logits(model, inputs):
    return model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, :].float()


@torch.inference_mode()
def predict(model, tokenizer, feedback: list[str], device: str) -> list[str]:
    inputs = batch_inputs(tokenizer, feedback, device)
    logits = next_logits(model, inputs)
    return [tuple(ROUTES)[idx] for idx in logits[:, token_ids(tokenizer)].argmax(-1).tolist()]


class Classifier:
    def __init__(self, adapter: Path, device: str):
        base, self.tokenizer = load_base(device)
        self.model = PeftModel.from_pretrained(base, adapter).eval()
        self.device = device

    def classify(self, feedback: str) -> str:
        return predict(self.model, self.tokenizer, [feedback], self.device)[0]

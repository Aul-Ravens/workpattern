import argparse
import json
import logging
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import torch

from .model import BUNDLED_ADAPTER, MAX_FEEDBACK, MODEL_DIR, ROOT, Classifier
from .playbook import ROUTES, SAMPLES, workflow

MODEL_NAME = "workpattern-triage"
PAGES_ORIGIN = "https://aul-ravens.github.io"
PAGES_API = {"/api/config": "GET", "/api/triage": "POST"}
HARNESS_FEEDBACK = re.compile(r"<workpattern-feedback>(.*?)</workpattern-feedback>", re.DOTALL)
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/images/robot-workshop.jpg": ("images/robot-workshop.jpg", "image/jpeg"),
    "/images/robot-review.jpg": ("images/robot-review.jpg", "image/jpeg"),
    "/images/robot-checklist.jpg": ("images/robot-checklist.jpg", "image/jpeg"),
    "/fonts/dm-sans.woff2": ("fonts/dm-sans.woff2", "font/woff2"),
    "/fonts/rubik.woff2": ("fonts/rubik.woff2", "font/woff2"),
}


def validate_feedback(value):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= MAX_FEEDBACK:
        raise ValueError(f"请输入 1 至 {MAX_FEEDBACK} 字的反馈。")
    return value.strip()


def markdown(result):
    steps = "\n".join(f"{n}. {step['title']}：{step['detail']}" for n, step in enumerate(result["steps"], 1))
    return f"## {result['title']}\n\n判断准则：{result['criterion']}\n\n{steps}"


def handler_for(model, adapter):
    inference_lock = Lock()
    report_path = adapter / "evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_bytes(self, body, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if self.path in PAGES_API:
                self.send_header("Vary", "Origin")
                if self.headers.get("Origin") == PAGES_ORIGIN:
                    self.send_header("Access-Control-Allow-Origin", PAGES_ORIGIN)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def fail(self, message, status):
            self.send_json({"error": {"message": message}}, status)

        def do_OPTIONS(self):
            method = PAGES_API.get(self.path)
            if (
                not method
                or self.headers.get("Origin") != PAGES_ORIGIN
                or self.headers.get("Access-Control-Request-Method") != method
            ):
                self.fail("此来源无法调用接口。", 403)
                return
            self.send_response(204)
            self.send_header("Vary", "Origin, Access-Control-Request-Method, Access-Control-Request-Private-Network")
            self.send_header("Access-Control-Allow-Origin", PAGES_ORIGIN)
            self.send_header("Access-Control-Allow-Methods", method)
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            if self.headers.get("Access-Control-Request-Private-Network") == "true":
                self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()

        def do_GET(self):
            if self.path in ASSETS:
                name, mime = ASSETS[self.path]
                self.send_bytes((ROOT / "web" / name).read_bytes(), mime)
            elif self.path == "/api/config":
                self.send_json({
                    "samples": SAMPLES, "routes": [workflow(code) for code in ROUTES],
                    "model": model.base_id, "device": model.device, "max_feedback": MAX_FEEDBACK,
                    "evaluation": {key: report[key] for key in ("eval_examples", "baseline", "lora")} if report else None,
                })
            elif self.path == "/v1/models":
                self.send_json({"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "workpattern"}]})
            else:
                self.fail("页面不存在。", 404)

        def do_POST(self):
            if self.path not in {"/api/triage", "/v1/chat/completions"}:
                self.fail("接口不存在。", 404)
                return
            origin = self.headers.get("Origin")
            local_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            pages_request = origin == PAGES_ORIGIN and self.path == "/api/triage"
            if origin and origin not in local_origins and not pages_request:
                self.fail("请从 WorkPattern 工作台提交反馈。", 403)
                return
            if pages_request and self.headers.get_content_type() != "application/json":
                self.fail("请求必须采用 application/json 格式。", 415)
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 131072:
                    raise ValueError("请求长度无效。")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("请求必须是 JSON 对象。")
                if self.path == "/api/triage":
                    feedback = validate_feedback(body.get("feedback"))
                else:
                    if body.get("model") != MODEL_NAME:
                        raise ValueError(f"请使用模型 {MODEL_NAME}。")
                    feedback = validate_feedback(self.last_user_message(body.get("messages")))
                started = time.perf_counter()
                with inference_lock:
                    route = model.classify(feedback)
                result = workflow(route)
            except (ValueError, UnicodeDecodeError) as error:
                self.fail(str(error), 400)
                return
            except Exception:
                logging.exception("Local inference failed")
                self.fail("本地模型推理失败，请检查服务日志。", 500)
                return

            if self.path == "/api/triage":
                self.send_json({"workflow": result, "elapsed_ms": round((time.perf_counter() - started) * 1000)})
            else:
                self.complete(markdown(result), body.get("stream") is True)

        @staticmethod
        def last_user_message(messages):
            if not isinstance(messages, list):
                raise ValueError("请提交 messages 数组。")
            user_texts = []
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, list):
                        content = "\n".join(
                            part["text"] for part in content
                            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                        )
                    if isinstance(content, str):
                        user_texts.append(content)
                        tagged = HARNESS_FEEDBACK.search(content)
                        if tagged:
                            return tagged.group(1)
            return user_texts[0] if user_texts else ""

        def complete(self, content, stream):
            meta = {"id": f"chatcmpl-{uuid.uuid4().hex}", "created": int(time.time()), "model": MODEL_NAME}
            if not stream:
                choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                self.send_json({**meta, "object": "chat.completion", "choices": [choice]})
                return
            chunks = [
                ({"role": "assistant", "content": content}, None),
                ({}, "stop"),
            ]
            events = []
            for delta, reason in chunks:
                choice = {"index": 0, "delta": delta, "finish_reason": reason}
                event = {**meta, "object": "chat.completion.chunk", "choices": [choice]}
                events.append(f"data: {json.dumps(event, ensure_ascii=False)}\n\n")
            self.send_bytes(("".join(events) + "data: [DONE]\n\n").encode(), "text/event-stream; charset=utf-8")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Run the local feedback workbench.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    local_adapter = MODEL_DIR / "adapter"
    default_adapter = local_adapter if (local_adapter / "adapter_config.json").is_file() else BUNDLED_ADAPTER
    adapter = args.adapter or default_adapter
    if not (adapter / "adapter_config.json").is_file():
        parser.error("Adapter missing. Run python -m workpattern.train or set --adapter.")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "cpu"
    model = Classifier(adapter, device)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(model, adapter)) as server:
        print(f"WorkPattern · http://127.0.0.1:{args.port} · {device}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

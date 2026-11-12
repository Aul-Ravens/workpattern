import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from torch.nn import functional as F

from .model import BASE_ID, BASE_MODELS, MAX_FEEDBACK, MODEL_DIR, ROOT, batch_inputs, load_base, next_logits, predict, token_ids
from .playbook import ROUTES

DATA = ROOT / "data"
DEFAULT_EVAL_FILES = [DATA / "eval.jsonl", DATA / "eval_bilingual.jsonl", DATA / "eval_release.jsonl"]
SEED = 42
CONTEXTS = [
    ("付费客户", "订单结算", "支付确认页长时间空白", "近一小时订单成功率下降七成", "账单预览", "对账时常需要反复查看明细"),
    ("仓管员", "库存盘点", "确认库存后提示系统错误", "三个仓库都无法提交当日盘点结果", "按货架筛选", "每次盘点都要手工找出本区商品"),
    ("企业管理员", "组织成员管理", "更新成员权限后页面崩溃", "多个租户无法给新员工分配工作权限", "按部门批量授权", "每月入职时逐个授权耗时较长"),
    ("商家", "商品发布", "上传图片后提交失败", "一半以上商家今天无法发布任何商品", "批量编辑商品标签", "他们每天都要为大量商品调整活动标记"),
    ("客服坐席", "会话接待", "接入用户后会话记录丢失", "多个班组的主要接待入口都已中断", "快速插入答复模板", "常见退款问题占到了每天大量接待时间"),
    ("门店店员", "优惠核销", "扫描后一直显示处理中", "整条连锁的收银柜台都无法完成核销", "按活动自动展示券", "每到活动期间找券的时间会明显增加"),
    ("财务同事", "发票入账", "提交时反复出现超时提示", "全部分公司的本月发票入账任务停滞", "按期间导出明细", "月末要逐页复制票据号到表格核对"),
    ("内容审核员", "内容复核", "打开待审内容时应用退出", "所有审核队列都积压且无法在备用端继续", "按语言分派任务", "每天要先阅读标题再手工把任务派给同事"),
    ("采购专员", "供货单确认", "签收确认写入了错误数量", "已有多笔实际收货被错误记账且还在新增", "批量对比历史价格", "续签前要手工打开十几张合同核价"),
    ("物流调度员", "车辆派单", "提交路线后无司机收到任务", "多数当日配送单积压且没有备用派单入口", "按片区复用路线", "节假日每次排班都要重新组合相同的站点"),
    ("课程运营", "课程报名", "提交报名后名额被重复扣减", "监控发现大量名额重复扣减且仍在发生", "按班级复制报名表", "每期开班需要重复录入相同的字段"),
    ("销售团队", "合同审批", "提交后看不到任何待办", "多个事业部合同无法送审且影响当天签约", "批量催办", "每周都要逐笔找到卡住的申请并发提醒"),
]


def synthetic_rows(include_followup=True):
    scenarios = json.loads((DATA / "train_scenarios.json").read_text(encoding="utf-8"))
    rows = []
    for label, templates in scenarios.items():
        for template in templates:
            for idx, context in enumerate(CONTEXTS, start=101):
                group, area, failure, impact, request, need = context
                feedback = template.format(
                    group=group, area=area, failure=failure, impact=impact,
                    request=request, need=need, number=idx,
                )
                rows.append({"route": label, "feedback": feedback})
    bilingual = json.loads((DATA / "bilingual_scenarios.json").read_text(encoding="utf-8"))
    for language, routes in bilingual["templates"].items():
        followup = bilingual["followup_templates"][language] if include_followup else {}
        for label, templates in routes.items():
            for ticket, context in enumerate(bilingual["contexts"], start=301):
                area, symptom, feature, need = context[language]
                for template in templates + followup.get(label, []):
                    feedback = template.format(
                        area=area, symptom=symptom, feature=feature, need=need, ticket=ticket,
                    )
                    rows.append({"route": label, "feedback": feedback})
    return rows


def read_jsonl(path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        feedback = row.get("feedback", "") if isinstance(row, dict) else ""
        if (
            not isinstance(feedback, str)
            or not 1 <= len(feedback.strip()) <= MAX_FEEDBACK
            or row.get("route") not in ROUTES
        ):
            raise ValueError(f"{path}:{line_number}: expected a feedback string and route A–E")
        rows.append({"feedback": feedback.strip(), "route": row["route"], "language": row.get("language")})
    if not rows:
        raise ValueError(f"{path} has no examples")
    return rows


def evaluate(model, tokenizer, rows, batch_size):
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        predictions.extend(predict(model, tokenizer, [row["feedback"] for row in batch], "cuda"))
    return predictions


def metrics(actual, predicted):
    pairs = list(zip(actual, predicted, strict=True))
    correct = sum(expected == chosen for expected, chosen in pairs)
    per_route = {}
    for label in ROUTES:
        tp = sum(expected == chosen == label for expected, chosen in pairs)
        support = actual.count(label)
        selected = predicted.count(label)
        per_route[label] = {
            "support": support, "correct": tp,
            "f1": round(2 * tp / (support + selected), 4) if support + selected else 0.0,
        }
    return {
        "correct": correct,
        "accuracy": round(correct / len(actual), 4),
        "macro_f1": round(sum(value["f1"] for value in per_route.values()) / len(ROUTES), 4),
        "per_route": per_route,
    }


def group_metrics(rows, predictions, field):
    groups = {}
    for index, row in enumerate(rows):
        if row.get(field):
            groups.setdefault(row[field], []).append(index)
    return {
        group: {
            "examples": len(indices),
            **{
                name: metrics([rows[i]["route"] for i in indices], [values[i] for i in indices])
                for name, values in predictions.items()
            },
        }
        for group, indices in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the feedback router with LoRA.")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--initial-pass", action="store_true", help="Use the initial synthetic training cases.")
    parser.add_argument("--eval-file", type=Path, action="append", help="Held-out JSONL file. Can be repeated.")
    parser.add_argument("--compare-adapter", type=Path, help="Existing LoRA adapter to evaluate on the same cases.")
    parser.add_argument("--init-adapter", type=Path, help="Continue training an adapter for the selected base model.")
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "adapter")
    parser.add_argument("--base-model", choices=BASE_MODELS, default=BASE_ID)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.init_adapter and PeftConfig.from_pretrained(args.init_adapter).base_model_name_or_path != args.base_model:
        parser.error("--init-adapter requires an adapter for the selected base model")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a CUDA GPU with BF16 support.")
    if args.train_file and args.initial_pass:
        parser.error("--initial-pass applies only to built-in training cases")
    rows = read_jsonl(args.train_file) if args.train_file else synthetic_rows(include_followup=not args.initial_pass)
    held_out = [
        {**row, "dataset": path.stem}
        for path in (args.eval_file or DEFAULT_EVAL_FILES)
        for row in read_jsonl(path)
    ]
    if {item["feedback"] for item in rows} & {item["feedback"] for item in held_out}:
        raise ValueError("Training and evaluation files contain identical feedback.")

    randomizer = random.Random(SEED)
    torch.manual_seed(SEED)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    predictions = {}
    previous_base_id = None
    if args.compare_adapter:
        previous_base_id = PeftConfig.from_pretrained(args.compare_adapter).base_model_name_or_path
        compare_base, compare_tokenizer = load_base("cuda", previous_base_id)
        previous = PeftModel.from_pretrained(compare_base, args.compare_adapter)
        predictions["previous"] = evaluate(previous.eval(), compare_tokenizer, held_out, args.batch_size)
        del previous, compare_base, compare_tokenizer
    base, tokenizer = load_base("cuda", args.base_model)
    predictions["baseline"] = evaluate(base.eval(), tokenizer, held_out, args.batch_size)
    if args.init_adapter:
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    targets = dict(zip(ROUTES, token_ids(tokenizer), strict=True))
    losses = []
    for epoch in range(args.epochs):
        model.train()
        randomizer.shuffle(rows)
        loss_total = 0.0
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            inputs = batch_inputs(tokenizer, [row["feedback"] for row in batch], "cuda")
            labels = torch.tensor([targets[row["route"]] for row in batch], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(next_logits(model, inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            loss_total += loss.item()
        losses.append(round(loss_total / math.ceil(len(rows) / args.batch_size), 4))
        print(f"epoch {epoch + 1}/{args.epochs}  loss={losses[-1]}", flush=True)

    predictions["lora"] = evaluate(model.eval(), tokenizer, held_out, args.batch_size)
    actual = [row["route"] for row in held_out]
    revision, _ = BASE_MODELS[args.base_model]
    model.peft_config["default"].base_model_name_or_path = args.base_model
    model.peft_config["default"].revision = revision
    model.save_pretrained(args.output)
    report = {
        "base_model": args.base_model, "revision": revision, "previous_base_model": previous_base_id,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__, "seed": SEED, "train_examples": len(rows),
        "eval_examples": len(held_out), "epochs": args.epochs, "batch_size": args.batch_size,
        "continued_training": args.init_adapter is not None, "learning_rate": args.learning_rate,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "peak_allocated_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "elapsed_seconds": round(time.perf_counter() - started, 1), "epoch_loss": losses,
        **{name: metrics(actual, values) for name, values in predictions.items()},
        "by_dataset": group_metrics(held_out, predictions, "dataset"),
        "by_language": group_metrics(held_out, predictions, "language"),
        "predictions": [
            {
                "dataset": row["dataset"], "language": row["language"], "expected": row["route"],
                **{name: values[index] for name, values in predictions.items()},
            }
            for index, row in enumerate(held_out)
        ],
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "eval_examples": len(held_out),
        **{name: report[name]["correct"] for name in predictions},
        "by_dataset": {
            key: {name: summary[name]["correct"] for name in predictions}
            for key, summary in report["by_dataset"].items()
        },
        "by_language": {
            key: {name: summary[name]["correct"] for name in predictions}
            for key, summary in report["by_language"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()

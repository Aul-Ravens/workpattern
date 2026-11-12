# WorkPattern · 反馈分流工作台

WorkPattern 帮产品运营和客服把一条用户反馈交到合适的处理流程。它用本地模型判断下一步，给出对应的执行清单。处理人可以改判、勾选已完成步骤，并导出 Markdown 记录。

[打开 GitHub Pages 工作台](https://aul-ravens.github.io/workpattern/)

## Local setup

Clone this repository and run these commands in PowerShell from the project directory. Use Python 3.13 on Windows.

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m workpattern.server
```

The first run downloads about 3.1 GB of base weights if they are missing. Open the workbench on the same computer, select **Connect local model**, and allow local network access if your browser asks. You can also use the local page. [Web workbench](https://aul-ravens.github.io/workpattern/#workbench) · [Local workbench](http://127.0.0.1:8765/)

After installation, double-click `start.cmd` to start the server. Pass `--device cpu` to use CPU inference.

## 产品场景

值班同学常常同时收到故障、已有缺陷和改进建议。相似描述可能需要完全不同的处理动作。例如，某个版本的导出功能异常可以先复现并建单，订单提交大面积失败时应先启动故障响应。WorkPattern 把产品负责人的判断准则写成带标签的历史案例，并把每种决定绑定到一份执行清单。

| 模型建议 | 触发条件 | 处理结果 |
| --- | --- | --- |
| 启动故障响应 | 核心任务大范围受阻，或存在资金、数据完整性风险 | 确认影响，交由值班负责人处置，验证恢复 |
| 复现并登记缺陷 | 异常可复现，影响局部或有备用入口 | 登记可复现条件、证据和回归结果 |
| 验证需求价值 | 正常功能需要改动，用户给出目标或场景 | 形成需求价值和最小交付方案的评审材料 |
| 关联已有工单 | 工单的触发条件和症状一致，影响未扩大 | 将证据和进展补充到原单 |
| 补齐反馈信息 | 现象、环境或用户目标不足以判断 | 收集信息，约定跟进时间后重新分流 |

工作台保留模型建议和复核结果。外部建单、通知和故障处置由处理人执行。

## 本机运行

已验证 Windows 原生环境，Python 3.13、PyTorch 2.7.1 CUDA 12.8 和 RTX 5070 Ti Laptop GPU。

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m workpattern.server
```

首次启动会下载约 3.1 GB 的基座权重，仓库内已附训练好的 LoRA 适配器。支持指定 `--device cpu` 进行 CPU 推理。Windows 已完成依赖安装时，可双击 `start.cmd` 启动。

在运行模型的同一台电脑上打开 [GitHub Pages 工作台](https://aul-ravens.github.io/workpattern/)，点击 **Connect local model**。如果浏览器询问是否允许站点访问本地网络，请选择允许。遇到浏览器限制时，可打开 [本机工作台](http://127.0.0.1:8765/)。

GitHub Pages 提供网页文件。页面把反馈直接发送到本机的 `127.0.0.1:8765`，模型和推理服务运行在本机。公开页面可以访问，执行分流需要在访问者自己的电脑上启动 WorkPattern。

如需读取外置模型，在启动前设置 `WORKPATTERN_MODEL_DIR` 为模型根目录，或在仓库根目录建立 `settings.local.json` 并填写 `model_dir`。1.5B 基座文件放在模型根目录的 `qwen-1.5b/`，0.5B 基座文件放在 `qwen/`。重新训练得到的适配器默认写入 `adapter/`。

## 训练与验证

当前模型基于 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)。LoRA 对注意力层做监督微调，学习从中英文反馈中选择五个处理流程之一。训练和评测均使用合成案例。将个人经验接入产品时，样本需要由对应负责人标注或复核。

```powershell
.venv/Scripts/python.exe -m workpattern.train --initial-pass --epochs 3 --learning-rate 0.0001 --output weights/initial-adapter
.venv/Scripts/python.exe -m workpattern.train --init-adapter weights/initial-adapter --epochs 1 --learning-rate 0.000035
.venv/Scripts/python.exe -m workpattern.train --train-file training.jsonl --eval-file validation.jsonl
```

自有训练文件采用 UTF-8 JSONL，字段为 `feedback` 和 `route`。分流标签和执行清单见 [playbook.py](workpattern/playbook.py)，样本来源见 [数据说明](data/README.md)。

| 同一提示词及候选标签 | 答对 | 准确率 |
| --- | ---: | ---: |
| Qwen2.5 1.5B 基座 | 53 / 85 | 62.4% |
| 上一版 0.5B LoRA | 70 / 85 | 82.4% |
| 当前 1.5B LoRA | 82 / 85 | 96.5% |

第一阶段使用 600 条样本训练 3 轮，第二阶段使用 660 条样本训练 1 轮。85 条评测反馈中，最后新增的 20 条复测反馈上，当前模型答对 18 条，上一版答对 15 条。当前模型在全部评测中仍漏判了两条涉及数据泄露的反馈。实际表现需要用团队的历史反馈复测。 [逐条预测与训练记录](model/adapter/evaluation.json)

## 接入 DeepSeek Harness

启动 WorkPattern 服务后，在另一个 PowerShell 窗口运行。需要 Node.js 24.9 或更高版本。

```powershell
$env:DSH_HOME = Join-Path (Get-Location).Path '.dsh'
$env:WORKPATTERN_HARNESS_KEY = 'local'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
'<workpattern-feedback>现有报表按设计运行。运营每周要逐个导出五个渠道报表，希望批量导出以便合并数据。</workpattern-feedback>' |
  npx --yes @deepseek-ai/dsh@0.2.2-rc.1 --profile headless --patch harness/workpattern.patch.yml --permission-mode read-only --approval-policy never -
```

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 通过本机 `/v1/chat/completions` 调用分流模型和流程清单。服务仅绑定回环地址，`local` 是客户端所需的占位值。本地工作台也可独立运行。

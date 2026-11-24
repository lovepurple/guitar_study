# 本地真人语音 TTS 安装与配置指南（MacBook Conda + Qwen3-TTS）

如果你觉得浏览器自带的语音播报机械感太强，可以在 MacBook（Apple Silicon M 系列芯片，24GB 内存即可流畅运行）上用 Conda 部署本地 **Qwen3-TTS-12Hz-1.7B-CustomVoice**，让 `CyberFret` 里的 AI 吉他老师用**接近真人的中文语音 + 语气指令**陪练，且完全免费、无需联网调用付费语音接口。

---

## 一、环境要求

- **硬件**：Apple Silicon Mac（M1/M2/M3/M4，建议 16GB+ 内存，利用 MPS 加速）或带 NVIDIA GPU 的 Windows/Linux PC
- **软件**：Miniconda / Anaconda（已安装 `conda`）
- **端口**：`38000`（TTS 服务）与 `8080`（`start_studio.py` 本地工作室）

---

## 二、全新机器从零安装步骤（MacBook Conda）

如果是一台还没有装过环境的 MacBook，按以下 4 步即可配好：

### 1. 创建 Python 3.11 环境
```bash
conda create -n py311 python=3.11 -y
conda activate py311
```

### 2. 安装依赖包
```bash
pip install torch fastapi uvicorn soundfile qwen-tts
```
> **避坑提示（国内网络离线加载）**：如果模型权重已经下载到 `~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/<hash>`，建议将脚本里的 `"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"` 直接替换为本地 snapshot 绝对路径，可避免启动时反复连接 `huggingface.co` 超时卡住。

### 3. 准备启动脚本（支持 MPS / CUDA / CPU 自动识别）
将如下代码保存为 `~/qwen-tts/qwen_tts_launcher.py`：

```python
import io
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

app = FastAPI()
model = None


def resolve_runtime_config():
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16, "CUDA"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32, "MPS"
    return "cpu", torch.float32, "CPU"


@app.on_event("startup")
async def load_model():
    global model
    device, dtype, backend = resolve_runtime_config()
    print(f"正在初始化 Qwen3-TTS ({backend}: {device}, dtype={dtype})...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=dtype,
    )
    print("Qwen3-TTS 已就绪！")


@app.get("/generate")
async def generate_speech(
    text: str = Query(..., description="要合成的文本"),
    speaker: str = Query("Vivian", description="预设音色名称"),
    instruct: str = Query(None, description="情感或语气指令"),
    lang: str = Query("Chinese", description="语言"),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    with torch.no_grad():
        wavs, sr = model.generate_custom_voice(
            text=text, language=lang, speaker=speaker, instruct=instruct
        )

    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format="WAV")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=38000)
```

### 4. 首次启动（会自动缓存模型权重）
```bash
conda activate py311
python ~/qwen-tts/qwen_tts_launcher.py
```
首次运行会自动下载 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` 到 `~/.cache/huggingface/hub/`（约 3.5GB），看到 `Qwen3-TTS 已就绪！` 及 `Uvicorn running on http://0.0.0.0:38000` 即表示启动成功。

---

## 三、在 MacBook 上日常启动与联调

### 1. 启动 MacBook 本地 TTS 服务
```bash
/Users/guangze.song/miniconda3/envs/py311/bin/python ~/qwen-tts/qwen_tts_launcher.py
```
（或先 `conda activate py311`，再 `python ~/qwen-tts/qwen_tts_launcher.py`）

### 2. 启动 CyberFret 本地工作室
- **如果你在 Windows 台式机练琴（通过局域网调用 MacBook 的 TTS）**：
  直接运行：
  ```bash
  python start_studio.py
  ```
  默认已将 `/tts/*` 代理到 `http://10.10.0.40:38000/*`。如果 MacBook 局域网 IP 变了，可通过环境变量覆盖：
  ```cmd
  set CYBERFRET_TTS_UPSTREAM=http://你的Mac局域网IP:38000
  python start_studio.py
  ```

- **如果你直接在 MacBook 上练琴**：
  ```bash
  CYBERFRET_TTS_UPSTREAM=http://127.0.0.1:38000 python3 start_studio.py
  ```

---

## 四、在网页里的配置与自动降级机制

打开 `http://127.0.0.1:8080/code_artifact.html` 后，点击 AI 老师右上角的 **「⚙️ 配置模型 / Key」**：
- **本地真人语音 TTS 服务**：默认预填 `http://10.10.0.40:38000`（经由 `start_studio.py` 的 `/tts/generate` 同源代理转发，无 CORS 跨域问题）
- **音色 (Speaker)**：默认 `Vivian`（也可换 `Ryan`、`Seraphina` 等模型支持的预设音色）
- **语气指令 (Instruct)**：默认 `用温柔耐心的电吉他老师口吻，语速适中`

> **自动降级保护**：如果 MacBook 上的 `qwen_tts_launcher.py` 当天没开机，网页会自动无缝回退到浏览器原生中文语音播报（`speechSynthesis`），不会报错卡死练习流程。

---

## 五、自检命令

在任意同局域网终端执行以下命令，确认返回 `HTTP:200` 且文件大小 > 10KB（即真实 WAV 音频）：
```bash
curl -s "http://10.10.0.40:38000/generate?text=你好，把手指立起来按弦&speaker=Vivian&lang=Chinese" --output test_tts.wav -w "HTTP:%{http_code} SIZE:%{size_download}\n"
```

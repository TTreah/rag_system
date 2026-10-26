"""
captioner.py — Qwen3-VL-Instruct 图像描述生成

用途：把入库的图片资产变成可检索、可引用的文字描述。
与 Qwen3-VL-Embedding 的分工：
    Embedding  只产出向量（图片本体用于跨模态检索）
    Instruct   看懂图片并生成文字（描述 / 测试题素材）

本地权重（ModelScope），MPS 推理，串行锁保护（MPS 非线程安全）。
"""

import os
import threading

_lock = threading.Lock()
_model = None
_processor = None


def _ensure_model(model_path: str):
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"[Captioner] Loading: {model_path} ...", flush=True)
    _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    _model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if torch.backends.mps.is_available():
        _model = _model.to("mps")
    _model.eval()
    print("[Captioner] Ready", flush=True)
    return _model, _processor


def caption_image(image_path: str, model_path: str,
                  prompt: str = None, max_new_tokens: int = 220) -> str:
    """生成图片描述。失败返回空串（调用方可退回旧描述）。"""
    if not os.path.exists(image_path):
        return ""
    model, processor = _ensure_model(model_path)
    import torch
    from qwen_vl_utils import process_vision_info

    prompt = prompt or (
        "请用中文客观描述这张图片：主体是什么、关键细节、场景，"
        "以及图中可见的文字（如有）。不要推测图片之外的信息。"
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with _lock, torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    # 去掉客套开头（"好的，这是对图片的客观描述：" 之类）
    lines = result.split("\n")
    if lines and len(lines[0]) < 60 and ("描述" in lines[0] or "图片" in lines[0]):
        result = "\n".join(lines[1:]).strip()
    return result


def unload():
    """释放模型（两阶段流水线中，描述阶段结束后让出内存给嵌入模型）。"""
    global _model, _processor
    _model = None
    _processor = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

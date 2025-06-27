from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, PretrainedConfig
import torch

model_path="/workspace/models/bge-m3"
config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )


model_kwargs = {}
model_kwargs.setdefault("torch_dtype", "float32")
model_xpu = SentenceTransformer(model_path,
                            model_kwargs=model_kwargs,
                            trust_remote_code=True)
# todo 1、切换到评估模式【关闭随机性层，确保评估结果稳定】；2、将模型移至GPU
# todo 2、推荐的写法
#  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#  model.to(device).eval()
model_xpu.eval().cuda()
prompts = [
        "Hello, my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ]
outputs = model_xpu.encode(prompts)
print("========================hf-xpu==========================")
for prompt, output in zip(prompts, outputs):
    embeds = output
    embeds_trimmed = ((str(embeds[:16])[:-1] +
                       ", ...]") if len(embeds) > 16 else embeds)
    print(f"Prompt: {prompt!r} | "
          f"Embeddings: {embeds_trimmed} (size={len(embeds)})")

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

for name, param in model_xpu.named_parameters():
    if "layer." in name and "layer.0" not in name:
        continue
    print(f"参数名: {name}")
    print(f"形状: {param.shape}")
    print(f"值:\n{param.data}")
    print("-" * 50)
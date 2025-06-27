from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# model_path = "/workspace/models/bge-m3"
# model = SentenceTransformer(model_path)  # 加载模型
# print(isinstance(model, SentenceTransformer))  # 输出 True
#
# model = CrossEncoder(model_path)  # 加载模型
# print(isinstance(model, CrossEncoder))  # 输出 True
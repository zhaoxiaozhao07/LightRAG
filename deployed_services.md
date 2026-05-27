# 已部署本地服务

## LLM模型

- model_name：qwen3.6-36b
- base_url: http://192.168.1.66:8000/v1
- apikey: 通过私有配置或 `LLM_BINDING_API_KEY` 设置，不写入文档

## VLM模型

- model_name：qwen3.6-36b-vision
- base_url: http://192.168.1.66:8000/v1
- apikey: 通过私有配置或 `VISION_BINDING_API_KEY` 设置，不写入文档
- 配置项：`vision_model`、`vision_base_url`、`vision_api_key`；未设置 endpoint/key 时默认沿用 LLM 的 `llm_base_url`、`llm_api_key`

## Embedding模型

- model_url: http://192.168.110.244:8002/v1
- api_key: 通过私有配置或 `EMBEDDING_BINDING_API_KEY` 设置，不写入文档
- model_name：bge-m3
- embedding_dim: 1024

## Rerank模型

- model_url: http://192.168.110.244:8003/v1/rerank
- model_name: bge-reranker
- api_key: 通过私有配置或 `RERANK_API_KEY` 设置，不写入文档

## 向量数据库

- milvus_uri: http://192.168.1.66:19530

## MinerU文档解析

- vlm_url: http://192.168.1.66:8001
- model_name: MinerU2.5-Pro-2B
- backend：vlm-http-client

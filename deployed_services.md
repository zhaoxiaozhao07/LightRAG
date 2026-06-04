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

## PostgreSQL 数据库

| 参数       | 值                                                             |
| ---------- | -------------------------------------------------------------- |
| Host       | `192.168.1.66`                                               |
| Port       | `5433`                                                       |
| 用户名     | `admin`                                                      |
| 密码       | `123456`                                                     |
| 数据库名   | `knowledge_base`                                             |
| 连接字符串 | `postgresql://admin:123456@192.168.1.66:5433/knowledge_base` |

---

## MinIO 对象存储

| 参数         | 值                            |
| ------------ | ----------------------------- |
| API Endpoint | `http://192.168.1.66:19000` |
| Console 地址 | `http://192.168.1.66:19001` |
| Access Key   | `admin`                     |
| Secret Key   | `admin123`                  |
| 是否使用 SSL | `False`                     |


## Neo4j 数据库
| 配置项         | 值                         |
| ------------   | -----                      |
| Neo4j URI      | `bolt://192.168.1.66:7687` |
| Neo4j 用户名   | `neo4j`                    |
| Neo4j 密码     | `LightRag@2026`            |


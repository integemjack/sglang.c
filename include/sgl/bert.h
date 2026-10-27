/* sgl/bert.h — BERT / RoBERTa 类编码器：embedding + rerank（cross-encoder）
 * 对应 sglang srt/models/bert.py（BertEmbeddingModel / BertForSequenceClassification）
 * 与 embedding_gemma2.py 之外的「编码器类」模型：无 KV cache、整段前向取 pooled 表示。
 *
 * 支持：
 *   - config.json: model_type bert / roberta（position_ids 语义：bert=0..n-1，
 *     roberta=i+padding_idx+1），absolute position embedding，post-LN 结构
 *     （HF BertModel / RobertaModel 的 encoder.layer.{i}.attention/output 布局），
 *     hidden_act = gelu(erf) / gelu_new(tanh) / relu / silu / linear
 *   - 权重前缀自动探测："" / "bert." / "roberta." / "model."
 *   - tokenizer.json: WordPiece（BertNormalizer + BertPreTokenizer）
 *   - pooling: mean（含 [CLS]/[SEP]，无 padding 时等价 sentence-transformers 的
 *     attention_mask 均值）/ cls，均做 L2 normalize
 */
#ifndef SGL_BERT_H
#define SGL_BERT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sgl_bert sgl_bert;

/* pooling 选择（sgl_bert_encode 的 pooling 参数） */
#define SGL_BERT_POOL_MEAN 0
#define SGL_BERT_POOL_CLS 1

/* 加载 HF BERT 目录（config.json + tokenizer.json + model.safetensors）。
 * 失败返回 NULL（原因走 SGL_ERROR 日志）。 */
sgl_bert *sgl_bert_load(const char *dir);
void sgl_bert_free(sgl_bert *m);
/* hidden_size（embedding 维度）；m 为 NULL 时返回 0 */
int sgl_bert_hidden(const sgl_bert *m);

/* 编码一段文本 -> embedding（pooling: 0=mean, 1=cls；再 L2 normalize）。
 * out_cap 必须是 >= sgl_bert_hidden(m)。成功返回 0，失败 -1。 */
int sgl_bert_encode(sgl_bert *m, const char *text, int pooling, float *out,
                    int out_cap);

/* rerank 打分（cross-encoder）：[CLS] q [SEP] d [SEP]（roberta 用双 [SEP]），
 * 取 [CLS] 过分类头。有 1 个 label 时返回该 logit；>=2 个 label 时返回 logit[1]
 * （NLI 正类/entailment）；无分类头时退化为 query/doc 两个 [CLS] 向量的余弦相似度。
 * 失败返回 NAN。 */
float sgl_bert_rerank(sgl_bert *m, const char *query, const char *doc);

/* tokenizer 单独可用：WordPiece 编码（含 [CLS]/[SEP]、按 max_len 截断；单序列不分批
 * 所以不产出 padding）。返回 token 数（含 [CLS]/[SEP]），失败 -1。
 * 内部按 dir 缓存已加载的 tokenizer（最多 4 个目录，非线程安全）。 */
int sgl_wordpiece_encode(const char *dir, const char *text, int *ids, int cap);

/* ---- 下面的 WordPiece 接口是 bert.c 与 wordpiece.c 的内部约定（也便于单独测试） ---- */
typedef struct sgl_wp sgl_wp;
sgl_wp *sgl_wp_load_dir(const char *dir); /* 读 dir/tokenizer.json */
void sgl_wp_free(sgl_wp *t);
int sgl_wp_vocab_size(const sgl_wp *t);
int sgl_wp_id(const sgl_wp *t, const char *piece); /* 查词表，-1 不存在 */
int sgl_wp_cls_id(const sgl_wp *t);
int sgl_wp_sep_id(const sgl_wp *t);
int sgl_wp_max_len(const sgl_wp *t); /* tokenizer.json 的 truncation max_length（dflt 512）*/
/* 不带 [CLS]/[SEP] 的纯 WordPiece id（未截断到 max_len，只受 cap 限制）*/
int sgl_wp_encode_plain(const sgl_wp *t, const char *text, int *ids, int cap);
/* 带 [CLS] ... [SEP] 并按 max_len 截断 */
int sgl_wp_encode_ids(const sgl_wp *t, const char *text, int *ids, int cap);

#ifdef __cplusplus
}
#endif

#endif
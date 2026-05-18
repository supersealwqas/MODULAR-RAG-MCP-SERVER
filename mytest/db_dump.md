# ChromaDB 数据库导出

导出时间: 2026-05-18 22:02:29

---

## 集合: `default`

- **chunk 总数**: 12
- **存储字段**: id, embedding, document, metadata

- **包含图片的 chunk**: 5/12

---

### Chunk 1: `079d78bd9a2808bb_0000_c48d6f3e`

| 字段 | 值 |
|------|-----|
| chunk_index | 0 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | 1. 整体结构与时间轴 |
| refined_by | rule |
| enriched_by | rule |

**tags**: llm, gpt, 模型, google, openai, meta, 开源, llama

**summary**: LLM背景知识介绍 学习⽬标 了解LLM背景的知识 掌握什么是语⾔模型 1 ⼤语⾔模型 (LLM) 背景 ⼤语⾔模型 (英⽂：Large Language Model，缩写LLM) 是⼀种⼈⼯智能模型, 旨在理解和⽣成⼈类语⾔.⼤语⾔模型可以处理多种⾃然语⾔任务，如⽂本分类、问答、翻译、对话等等.

**图片引用** (1 张):

| image_id | page | 尺寸 | text_offset | 文件路径 |
|----------|------|------|-------------|----------|
| `079d78bd9a2808bb_1_0` | p1 | 661×520 | 524 | `079d78bd9a2808bb_1_0.png` |

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "file_size": 1786838,
  "doc_hash": "079d78bd9a2808bb",
  "images": [
    {
      "id": "079d78bd9a2808bb_1_0",
      "path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\images\\default\\079d78bd9a2808bb\\079d78bd9a2808bb_1_0.png",
      "page": 1,
      "text_offset": 524,
      "text_length": 29,
      "position": {
        "xref": 7,
        "width": 661,
        "height": 520
      }
    }
  ],
  "tags": [
    "llm",
    "gpt",
    "模型",
    "google",
    "openai",
    "meta",
    "开源",
    "llama"
  ],
  "image_captions": {
    "079d78bd9a2808bb_1_0": "这张图是**大语言模型（LLM）的进化树**，以**时间轴（2018–2023年）**为纵轴，清晰展示模型的**演化关系**、**开源/闭源分支**、**研发方**及**技术发展脉络**，适合文档检索时快速定位模型演进逻辑。\n\n\n### 1. 整体结构与时间轴  \n纵轴从下到上标注年份（2018→2023），横轴以**蓝色主分支（闭源模型）**和**粉色分支（开源模型）**为核心，从底层早期嵌入模型（如Word2Vec、ELMo）向上分化、演化。  \n\n\n### 2. 模型节点与演化关系（按年份/分支）  \n#### （1）早期基础（2018年）  \n底层节点为嵌入模型：**ELMo**（Allen AI）、**Word2Vec**、**FastText**，作为LLM的技术基础。  \n\n#### （2）预训练模型萌芽（2019–2020年）  \n- **开源分支（粉色）**：BERT（Google）、RoBERTa（Facebook）、ALBERT（Google）、ELECTRA（Google）、T5（Google）、UniLM（Microsoft）等，开启“预训练+微调”范式。  \n- **闭源分支（蓝色）**：GPT-1（OpenAI）分化出早期闭源路线。  \n\n#### （3）大模型爆发（2021–2023年）  \n- **闭源分支（蓝色）**：GPT-2/GPT-3（OpenAI）、Codex（OpenAI）、InstructGPT（OpenAI）、Jurassic-1（AI21 Labs）、GPT-4（OpenAI）、Claude（Anthropic）、Bard（Google）、Galactica（Meta）等，模型规模与能力快速提升。  \n- **开源分支（粉色）**：LLaMA（Meta）、OPT（Meta）、Bloom（BigScience）、LLaMA-2（Meta）、Switch Transformer（Google）等，开源社区/企业推动大模型普及。  \n\n\n### 3. 符号与图例（检索关键标识）  \n- **颜色**：粉色=**Open-Source（开源模型）**，蓝色=**Closed-Source（闭源模型）**（底部图例标注）。  \n- **符号**：研发方标志（如OpenAI的“O”、Google的“G”、Meta的“M”、Anthropic的“A”等），快速识别模型所属公司。  \n\n\n### 4. 技术内容与演化逻辑  \n进化树体现LLM发展的**三大维度**：  \n- **时间线**：从嵌入模型（2018）→ 预训练模型（2019–2020）→ 大语言模型（2021–2023），规模/能力持续突破。  \n- **开源vs闭源**：粉色分支（如BERT、LLaMA）代表开源生态，蓝色分支（如GPT、Claude）代表闭源（API/私有化）路线，体现技术路线的差异化竞争。  \n- **研发方**：Google、OpenAI、Meta、Anthropic、Baidu等多家巨头/初创公司参与，通过符号标注，便于定位特定企业的模型发展。  \n\n\n### 文档检索价值  \n该图可快速辅助检索：  \n- 特定年份的模型（如“2023年有哪些LLM”）；  \n- 特定公司的模型（如“Google的LLM演化”）；  \n- 开源/闭源模型的分支脉络（如“开源LLM的发展路径”）；  \n- 模型间的演化关系（如“GPT系列的进化”）。  \n\n\n（注：图中模型名称、年份、研发方标志等均为LLM领域的核心检索关键词，结合时间轴和分支结构，可高效梳理技术发展脉络。）"
  },
  "summary": "LLM背景知识介绍 学习⽬标 了解LLM背景的知识 掌握什么是语⾔模型 1 ⼤语⾔模型 (LLM) 背景 ⼤语⾔模型 (英⽂：Large Language Model，缩写LLM) 是⼀种⼈⼯智能模型, 旨在理解和⽣成⼈类语⾔.⼤语⾔模型可以处理多种⾃然语⾔任务，如⽂本分类、问答、翻译、对话等等.",
  "image_refs": [
    "079d78bd9a2808bb_1_0"
  ],
  "refined_by": "rule",
  "enriched_by": "rule",
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf",
  "chunk_index": 0,
  "title": "1. 整体结构与时间轴",
  "doc_type": "pdf"
}
```

</details>

**文本内容** (2399 字符):

```
LLM背景知识介绍

学习⽬标

了解LLM背景的知识
掌握什么是语⾔模型

1 ⼤语⾔模型 (LLM) 背景

⼤语⾔模型 (英⽂：Large Language Model，缩写LLM) 是⼀种⼈⼯智能模型, 旨在理解和⽣成⼈类语⾔.
⼤语⾔模型可以处理多种⾃然语⾔任务，如⽂本分类、问答、翻译、对话等等.

通常, ⼤语⾔模型 (LLM) 是指包含数千亿 (或更多) 参数的语⾔模型(⽬前定义参数量超过10B的模型为⼤
语⾔模型)，这些参数是在⼤量⽂本数据上训练的，例如模型 GPT-3、ChatGPT、PaLM、BLOOM和
LLaMA等.

截⽌23年，语⾔模型发展⾛过了三个阶段：

第⼀阶段 ：设计⼀系列的⾃监督训练⽬标（MLM、NSP等），设计新颖的模型架构
（Transformer），遵循Pre-training和Fine-tuning范式。典型代表是BERT、GPT、XLNet等；
第⼆阶段 ：逐步扩⼤模型参数和训练语料规模，探索不同类型的架构。典型代表是BART、T5、
GPT-3等；
第三阶段 ：⾛向AIGC（Artiﬁcial Intelligent Generated Content）时代，模型参数规模步⼊千万

这张图是**大语言模型（LLM）的进化树**，以**时间轴（2018–2023年）**为纵轴，清晰展示模型的**演化关系**、**开源/闭源分支**、**研发方**及**技术发展脉络**，适合文档检索时快速定位模型演进逻辑。


### 1. 整体结构与时间轴  
纵轴从下到上标注年份（2018→2023），横轴以**蓝色主分支（闭源模型）**和**粉色分支（开源模型）**为核心，从底层早期嵌入模型（如Word2Vec、ELMo）向上分化、演化。  


### 2. 模型节点与演化关系（按年份/分支）  
#### （1）早期基础（2018年）  
底层节点为嵌入模型：**ELMo**（Allen AI）、**Word2Vec**、**FastText**，作为LLM的技术基础。  

#### （2）预训练模型萌芽（2019–2020年）  
- **开源分支（粉色）**：BERT（Google）、RoBERTa（Facebook）、ALBERT（Google）、ELECTRA（Google）、T5（Google）、UniLM（Microsoft）等，开启“预训练+微调”范式。  
- **闭源分支（蓝色）**：GPT-1（OpenAI）分化出早期闭源路线。  

#### （3）大模型爆发（2021–2023年）  
- **闭源分支（蓝色）**：GPT-2/GPT-3（OpenAI）、Codex（OpenAI）、InstructGPT（OpenAI）、Jurassic-1（AI21 Labs）、GPT-4（OpenAI）、Claude（Anthropic）、Bard（Google）、Galactica（Meta）等，模型规模与能力快速提升。  
- **开源分支（粉色）**：LLaMA（Meta）、OPT（Meta）、Bloom（BigScience）、LLaMA-2（Meta）、Switch Transformer（Google）等，开源社区/企业推动大模型普及。  


### 3. 符号与图例（检索关键标识）  
- **颜色**：粉色=**Open-Source（开源模型）**，蓝色=**Closed-Source（闭源模型）**（底部图例标注）。  
- **符号**：研发方标志（如OpenAI的“O”、Google的“G”、Meta的“M”、Anthropic的“A”等），快速识别模型所属公司。  


### 4. 技术内容与演化逻辑  
进化树体现LLM发展的**三大维度**：  
- **时间线**：从嵌入模型（2018）→ 预训练模型（2019–2020）→ 大语言模型（2021–2023），规模/能力持续突破。  
- **开源vs闭源**：粉色分支（如BERT、LLaMA）代表开源生态，蓝色分支（如GPT、Claude）代表闭源（API/私有化）路线，体现技术路线的差异化竞争。  
- **研发方**：Google、OpenAI、Meta、Anthropic、Baidu等多家巨头/初创公司参与，通过符号标注，便于定位特定企业的模型发展。  


### 文档检索价值  
该图可快速辅助检索：  
- 特定年份的模型（如“2023年有哪些LLM”）；  
- 特定公司的模型（如“Google的LLM演化”）；  
- 开源/闭源模型的分支脉络（如“开源LLM的发展路径”）；  
- 模型间的演化关系（如“GPT系列的进化”）。  


（注：图中模型名称、年份、研发方标志等均为LLM领域的核心检索关键词，结合时间轴和分支结构，可高效梳理技术发展脉络。）

亿，模型架构为⾃回归架构，⼤模型⾛向对话式、⽣成式、多模态时代，更加注重与⼈类交互进⾏

对⻬，实现可靠、安全、⽆毒的模型。典型代表是InstructionGPT、ChatGPT、Bard、GPT-4等。

2 语⾔模型 (Language Model, LM)

语⾔模型（Language Model）旨在建模词汇序列的⽣成概率，提升机器的语⾔智能⽔平，使机器能够
模拟⼈类说话、写作的模式进⾏⾃动⽂本输出。

通俗理解: ⽤来计算⼀个句⼦的概率的模型，也就是判断⼀句话是否是⼈话的概率.

标准定义：对于某个句⼦序列, 如S = {W1, W2, W3, …, Wn}, 语⾔模型就是计算该序列发⽣的概率, 即
P(S). 如果给定的词序列符合语⽤习惯, 则给出⾼概率, 否则给出低概率.

举例说明：
```

---

### Chunk 2: `079d78bd9a2808bb_0001_1f1cc7bb`

| 字段 | 值 |
|------|-----|
| chunk_index | 1 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | 通俗理解: ⽤来计算⼀个句⼦的概率的模型，也就是判断⼀句话是否是⼈话的概率. |
| refined_by | rule |
| enriched_by | rule |

**tags**: 模型, 个句, 基于规则和统, 计的语, gram, 通俗理解, 来计算, 的概率的模型

**summary**: 通俗理解: ⽤来计算⼀个句⼦的概率的模型，也就是判断⼀句话是否是⼈话的概率.标准定义：对于某个句⼦序列, 如S = {W1, W2, W3, …, Wn}, 语⾔模型就是计算该序列发⽣的概率, 即 P(S).如果给定的词序列符合语⽤习惯, 则给出⾼概率, 否则给出低概率.

**图片**: 无

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "doc_hash": "079d78bd9a2808bb",
  "refined_by": "rule",
  "title": "通俗理解: ⽤来计算⼀个句⼦的概率的模型，也就是判断⼀句话是否是⼈话的概率.",
  "tags": [
    "模型",
    "个句",
    "基于规则和统",
    "计的语",
    "gram",
    "通俗理解",
    "来计算",
    "的概率的模型"
  ],
  "doc_type": "pdf",
  "enriched_by": "rule",
  "summary": "通俗理解: ⽤来计算⼀个句⼦的概率的模型，也就是判断⼀句话是否是⼈话的概率.标准定义：对于某个句⼦序列, 如S = {W1, W2, W3, …, Wn}, 语⾔模型就是计算该序列发⽣的概率, 即 P(S).如果给定的词序列符合语⽤习惯, 则给出⾼概率, 否则给出低概率.",
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf",
  "file_size": 1786838,
  "chunk_index": 1
}
```

</details>

**文本内容** (847 字符):

```
通俗理解: ⽤来计算⼀个句⼦的概率的模型，也就是判断⼀句话是否是⼈话的概率.

标准定义：对于某个句⼦序列, 如S = {W1, W2, W3, …, Wn}, 语⾔模型就是计算该序列发⽣的概率, 即
P(S). 如果给定的词序列符合语⽤习惯, 则给出⾼概率, 否则给出低概率.

举例说明：

假设我们要为中⽂创建⼀个语⾔模型， 表示词典， ={⿊⻢、程序、员、来、学习}， 属于
。语⾔模型描述：给定词典 , 能够计算出任意单词序列
率

是⼀句话的概

, 其中

那么如何计算⼀个句⼦的

呢？最简单的⽅法就是计数，假设数据集中共有 个句⼦，我们可

以统计⼀下数据集中

每个句⼦出现的次数，如果假设为 ，则

. 那么可以想象⼀下，这个模型的预测能⼒⼏乎为0，⼀旦单词序列没在之前数据集中

出现过，模型的输出概率就是0，显然相当不合理。
我们可以根据概率论中的链式法则，将 可以表示为：

如果能计算

，那么就能轻松得到

中，我们也可以看到语⾔模型的另外⼀个定义：能够计算出

, 所以在某些⽂献
的模型就是语⾔模

型。

从⽂本⽣成⻆度，也可以这样定义语⾔模型：给定⼀个短语（⼀个词组或者⼀句话），语⾔模型可以⽣

成（预测）接下来的⼀个词。

基于语⾔模型技术的发展，可以将语⾔模型分为四种类型：

基于规则和统计的语⾔模型

神经语⾔模型

预训练语⾔模型

⼤语⾔模型

2.1 基于规则和统计的语⾔模型（N-gram）

由⼈⼯设计特征并使⽤统计⽅法对固定⻓度的⽂本窗⼝序列进⾏建模分析，这种建模⽅式也被称为N-
gram语⾔模型。在上述例⼦中计算句⼦序列概率我们使⽤链式法则计算， 该⽅法存在两个缺陷：

参数空间过⼤：条件概率

的可能性太多，⽆法估算，也不⼀定有⽤

数据稀疏严重：许多词对的组合，在语料库中都没有出现，依据最⼤似然估计得到的概率为0

为了解决上述问题，引⼊⻢尔科夫假设：随意⼀个词出现的概率只与它前⾯出现的有限的⼀个或者⼏个

词有关。
```

---

### Chunk 3: `079d78bd9a2808bb_0002_e068f0ce`

| 字段 | 值 |
|------|-----|
| chunk_index | 2 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | 表格结构与标识 |
| refined_by | rule |
| enriched_by | rule |

**tags**: 篮球, 晚饭, 食物, bigram, 那么我们就称, 之为, 词汇, 数值

**summary**: 参数空间过⼤：条件概率 的可能性太多，⽆法估算，也不⼀定有⽤ 数据稀疏严重：许多词对的组合，在语料库中都没有出现，依据最⼤似然估计得到的概率为0 为了解决上述问题，引⼊⻢尔科夫假设：随意⼀个词出现的概率只与它前⾯出现的有限的⼀个或者⼏个 词有关。如果⼀个词的出现与它周围的词是独⽴的，那么我们就称之为unigram也就是⼀元语⾔模型.

**图片引用** (2 张):

| image_id | page | 尺寸 | text_offset | 文件路径 |
|----------|------|------|-------------|----------|
| `079d78bd9a2808bb_3_1` | p3 | 1656×672 | 1958 | `079d78bd9a2808bb_3_1.png` |
| `079d78bd9a2808bb_3_2` | p3 | 1654×166 | 1989 | `079d78bd9a2808bb_3_2.png` |

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "title": "表格结构与标识",
  "image_refs": [
    "079d78bd9a2808bb_3_1",
    "079d78bd9a2808bb_3_2"
  ],
  "chunk_index": 2,
  "doc_hash": "079d78bd9a2808bb",
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf",
  "enriched_by": "rule",
  "tags": [
    "篮球",
    "晚饭",
    "食物",
    "bigram",
    "那么我们就称",
    "之为",
    "词汇",
    "数值"
  ],
  "images": [
    {
      "id": "079d78bd9a2808bb_3_1",
      "path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\images\\default\\079d78bd9a2808bb\\079d78bd9a2808bb_3_1.png",
      "page": 3,
      "text_offset": 1958,
      "text_length": 29,
      "position": {
        "xref": 21,
        "width": 1656,
        "height": 672
      }
    },
    {
      "id": "079d78bd9a2808bb_3_2",
      "path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\images\\default\\079d78bd9a2808bb\\079d78bd9a2808bb_3_2.png",
      "page": 3,
      "text_offset": 1989,
      "text_length": 29,
      "position": {
        "xref": 22,
        "width": 1654,
        "height": 166
      }
    }
  ],
  "image_captions": {
    "079d78bd9a2808bb_3_1": "这是一张**中文词汇转移频次统计矩阵表格**，属于自然语言处理（NLP）领域的基础统计图表，用于量化8个中文词汇在语料中的前后出现关联频次：\n\n### 表格结构与标识\n- 表头：蓝色背景，从左到右依次为词汇「我、想、去、打、篮球、晚饭、食物、喝」；\n- 左侧行标题：浅灰色背景，从上到下依次为同款8个词汇；\n- 单元格数值：代表**行标题词汇指向列标题词汇的出现频次**，即行词汇后紧邻列词汇的统计次数。\n\n### 核心数据特征\n1.  **高频关联路径**\n    全表最高频关联为：「我」→「想」800次，其次是「去」→「打」690次、「想」→「去」600次、「去」→「喝」100次，构成“我-想-去-打/喝”的高概率语义路径；\n    其他较高频次：「晚饭」→「去」30次、「打」→「篮球」20次、「我」→「喝」20次。\n\n2.  **低频/零频次区域**\n    - 名词「篮球」：所有从「篮球」出发的关联频次均为0，仅作为被动关联对象存在，仅「打」→「篮球」为20次；\n    - 词汇「食物」「喝」：仅存在极低的单向关联，「食物」→「去」3次、「喝」→「去」1次，其余方向均为0；\n    - 大量单元格为0值，反映这些词汇在语料中不存在前后紧邻的统计关系。\n\n该表格可支撑文本生成、语法模式识别等NLP任务的词汇概率计算。",
    "079d78bd9a2808bb_3_2": "这张图片展示了一个**两行八列的表格**，用于呈现**中文词汇（或单字）与对应数值**的映射关系，无流程图，核心是文字（中文词汇）与数字的对应结构。  \n\n\n### 表格细节（行、列、内容）：  \n- **第一行（表头）**：背景为蓝色，文字为白色，从左到右依次为8个中文词汇/单字：  \n  `我`、`想`、`去`、`打`、`篮球`、`晚饭`、`食物`、`喝`  \n\n- **第二行（数据）**：背景为浅灰色（或浅蓝色），文字为黑色，对应第一行各词汇的数值依次为：  \n  `2100`、`900`、`2000`、`800`、`1000`、`120`、`300`、`260`  \n\n\n### 技术内容推测（结合场景）：  \n该表格可能用于**中文自然语言处理**（如文本分析、词汇权重统计、字符计数）或**输入法/语音识别**等技术场景，通过数值量化中文词汇的某种属性（如词频、向量维度、权重值等）。表格清晰呈现了8个中文词汇（单字/词）与其对应数值的映射，无流程图，核心是“词汇-数值”的结构化关系。  \n\n\n### 总结（文档检索友好）：  \n这是一张**中文词汇与数值映射的两行八列表格**，第一行列标题为中文词汇（我、想、去、打、篮球、晚饭、食物、喝），第二行为对应数值（2100、900、2000、800、1000、120、300、260），用于展示中文词汇的量化属性（如词频、权重），无流程图或其他图形，核心信息为“词汇-数值”的对应关系。"
  },
  "refined_by": "rule",
  "file_size": 1786838,
  "doc_type": "pdf",
  "summary": "参数空间过⼤：条件概率 的可能性太多，⽆法估算，也不⼀定有⽤ 数据稀疏严重：许多词对的组合，在语料库中都没有出现，依据最⼤似然估计得到的概率为0 为了解决上述问题，引⼊⻢尔科夫假设：随意⼀个词出现的概率只与它前⾯出现的有限的⼀个或者⼏个 词有关。如果⼀个词的出现与它周围的词是独⽴的，那么我们就称之为unigram也就是⼀元语⾔模型."
}
```

</details>

**文本内容** (1965 字符):

```
参数空间过⼤：条件概率

的可能性太多，⽆法估算，也不⼀定有⽤

数据稀疏严重：许多词对的组合，在语料库中都没有出现，依据最⼤似然估计得到的概率为0

为了解决上述问题，引⼊⻢尔科夫假设：随意⼀个词出现的概率只与它前⾯出现的有限的⼀个或者⼏个

词有关。

如果⼀个词的出现与它周围的词是独⽴的，那么我们就称之为unigram也就是⼀元语⾔模型.

如果⼀个词的出现仅依赖于它前⾯出现的⼀个词，那么我们就称之为bigram.

如果⼀个词的出现仅依赖于它前⾯出现的两个词，那么我们就称之为trigram.

⼀般来说，N元模型就是假设当前词的出现概率只与它前⾯的N-1个词有关，⽽这些概率参数都是
可以通过⼤规模语料库来计算，⽐如三元概率：

在实践中⽤的最多的就是bigram和trigram，接下来以bigram语⾔模型为例，理解其⼯作原理：

⾸先我们准备⼀个语料库（简单理解让模型学习的数据集），为了计算对应的⼆元模型的参数，即

，我们要先计数即

 计数结果如下：

，然后计数

 , 再⽤除法可得到概率。

 的计数结果如下:

这是一张**中文词汇转移频次统计矩阵表格**，属于自然语言处理（NLP）领域的基础统计图表，用于量化8个中文词汇在语料中的前后出现关联频次：

### 表格结构与标识
- 表头：蓝色背景，从左到右依次为词汇「我、想、去、打、篮球、晚饭、食物、喝」；
- 左侧行标题：浅灰色背景，从上到下依次为同款8个词汇；
- 单元格数值：代表**行标题词汇指向列标题词汇的出现频次**，即行词汇后紧邻列词汇的统计次数。

### 核心数据特征
1.  **高频关联路径**
    全表最高频关联为：「我」→「想」800次，其次是「去」→「打」690次、「想」→「去」600次、「去」→「喝」100次，构成“我-想-去-打/喝”的高概率语义路径；
    其他较高频次：「晚饭」→「去」30次、「打」→「篮球」20次、「我」→「喝」20次。

2.  **低频/零频次区域**
    - 名词「篮球」：所有从「篮球」出发的关联频次均为0，仅作为被动关联对象存在，仅「打」→「篮球」为20次；
    - 词汇「食物」「喝」：仅存在极低的单向关联，「食物」→「去」3次、「喝」→「去」1次，其余方向均为0；
    - 大量单元格为0值，反映这些词汇在语料中不存在前后紧邻的统计关系。

该表格可支撑文本生成、语法模式识别等NLP任务的词汇概率计算。

这张图片展示了一个**两行八列的表格**，用于呈现**中文词汇（或单字）与对应数值**的映射关系，无流程图，核心是文字（中文词汇）与数字的对应结构。  


### 表格细节（行、列、内容）：  
- **第一行（表头）**：背景为蓝色，文字为白色，从左到右依次为8个中文词汇/单字：  
  `我`、`想`、`去`、`打`、`篮球`、`晚饭`、`食物`、`喝`  

- **第二行（数据）**：背景为浅灰色（或浅蓝色），文字为黑色，对应第一行各词汇的数值依次为：  
  `2100`、`900`、`2000`、`800`、`1000`、`120`、`300`、`260`  


### 技术内容推测（结合场景）：  
该表格可能用于**中文自然语言处理**（如文本分析、词汇权重统计、字符计数）或**输入法/语音识别**等技术场景，通过数值量化中文词汇的某种属性（如词频、向量维度、权重值等）。表格清晰呈现了8个中文词汇（单字/词）与其对应数值的映射，无流程图，核心是“词汇-数值”的结构化关系。  


### 总结（文档检索友好）：  
这是一张**中文词汇与数值映射的两行八列表格**，第一行列标题为中文词汇（我、想、去、打、篮球、晚饭、食物、喝），第二行为对应数值（2100、900、2000、800、1000、120、300、260），用于展示中文词汇的量化属性（如词频、权重），无流程图或其他图形，核心信息为“词汇-数值”的对应关系。

那么bigram语⾔模型针对上述语料的参数计算结果如何实现？假如，我想计算 ! " #

 ,

计算过程如下显示：（其他参数计算过程类似）

! " #

# !

#

如果针对这个语料库的⼆元模型（bigram）建⽴好之后，就可以实现我们的⽬标计算。
计算⼀个句⼦的概率，举例如下：

# ! $ % & '

! " #

$ " !

% " $

& ' " %

预测⼀句话最可能出现的下⼀个词汇，⽐如：我想去打【mask】? 思考：mask = 篮球 或者 mask
= 晚饭。

# ! $ % & '

# ! $ % ( )

可以看出 # ! $ % & '
```

---

### Chunk 4: `079d78bd9a2808bb_0003_e4290089`

| 字段 | 值 |
|------|-----|
| chunk_index | 3 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | ! $ % & ' |
| refined_by | rule |
| enriched_by | rule |

**tags**: 神经, mask, gram, 篮球, 优点, 缺点, 个词, 络的第

**summary**: " # $ " !% " $ & ' " % 预测⼀句话最可能出现的下⼀个词汇，⽐如：我想去打【mask】?思考：mask = 篮球 或者 mask = 晚饭。

**图片**: 无

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "file_size": 1786838,
  "chunk_index": 3,
  "doc_type": "pdf",
  "tags": [
    "神经",
    "mask",
    "gram",
    "篮球",
    "优点",
    "缺点",
    "个词",
    "络的第"
  ],
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf",
  "enriched_by": "rule",
  "refined_by": "rule",
  "title": "! $ % & '",
  "summary": "\" # $ \" !% \" $ & ' \" % 预测⼀句话最可能出现的下⼀个词汇，⽐如：我想去打【mask】?思考：mask = 篮球 或者 mask = 晚饭。",
  "doc_hash": "079d78bd9a2808bb"
}
```

</details>

**文本内容** (869 字符):

```
# ! $ % & '

! " #

$ " !

% " $

& ' " %

预测⼀句话最可能出现的下⼀个词汇，⽐如：我想去打【mask】? 思考：mask = 篮球 或者 mask
= 晚饭。

# ! $ % & '

# ! $ % ( )

可以看出 # ! $ % & '

# ! $ % ( ) ，因此mask = 篮球，对⽐真实语境下，也符合⼈类习

惯。

N-gram语⾔模型的特点：

优点：采⽤极⼤似然估计, 参数易训练; 完全包含了前n-1个词的全部信息; 可解释性强, 直观易理
解。

缺点：缺乏⻓期以来，只能建模到前n-1个词; 随着n的增⼤，参数空间呈指数增⻓；数据稀疏，难
免会出现OOV问题; 单纯的基于统计频次，泛化能⼒差.

2.2 神经⽹络语⾔模型

伴随着神经⽹络技术的发展，⼈们开始尝试使⽤神经⽹络来建⽴语⾔模型进⽽解决N-gram语⾔模型存
在的问题。

上图属于⼀个最基础的神经⽹络架构：

模型的输⼊：

就是前n-1个词。现在需要根据这已知的n-1个词预测下⼀个

词 。

表示 所对应的词向量.

⽹络的第⼀层（输⼊层）是将

⼀个

⼤⼩的向量，记作 .

这n-1个向量⾸尾拼接起来形成

⽹络的第⼆层（隐藏层）就如同普通的神经⽹络，直接使⽤⼀个全连接层, 通过全连接层后再使⽤

这个激活函数进⾏处理。

 代表语料的词汇)，本质上这个输出层也是⼀个全连接
⽹络的第三层（输出层）⼀共有 个节点 (
层。每个输出节点 表示下⼀个词语为 的未归⼀化log 概率。最后使⽤ softmax 激活函数将输出
值 进⾏归⼀化。得到最⼤概率值，就是我们需要预测的结果。

神经⽹络特点：

优点：利⽤神经⽹络去建模当前词出现的概率与其前 n-1 个词之间的约束关系，很显然这种⽅式相
⽐ n-gram 具有更好的泛化能⼒，只要词表征⾜够好。从⽽很⼤程度地降低了数据稀疏带来的问
题。

缺点：对⻓序列的建模能⼒有限，可能会出现⻓距离遗忘以及训练时的梯度消失等问题，构建的模
```

---

### Chunk 5: `079d78bd9a2808bb_0004_3a1841c1`

| 字段 | 值 |
|------|-----|
| chunk_index | 4 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | 整体结构 |
| refined_by | rule |
| enriched_by | rule |

**tags**: gpt, transformer, index, 模型, tanh, softmax, 神经, 优点

**summary**: 神经⽹络特点： 优点：利⽤神经⽹络去建模当前词出现的概率与其前 n-1 个词之间的约束关系，很显然这种⽅式相 ⽐ n-gram 具有更好的泛化能⼒，只要词表征⾜够好。从⽽很⼤程度地降低了数据稀疏带来的问 题。缺点：对⻓序列的建模能⼒有限，可能会出现⻓距离遗忘以及训练时的梯度消失等问题，构建的模 型难以进⾏稳定的⻓⽂本输出。

**图片引用** (1 张):

| image_id | page | 尺寸 | text_offset | 文件路径 |
|----------|------|------|-------------|----------|
| `079d78bd9a2808bb_5_3` | p5 | 1236×1036 | 3585 | `079d78bd9a2808bb_5_3.png` |

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "images": [
    {
      "id": "079d78bd9a2808bb_5_3",
      "path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\images\\default\\079d78bd9a2808bb\\079d78bd9a2808bb_5_3.png",
      "page": 5,
      "text_offset": 3585,
      "text_length": 29,
      "position": {
        "xref": 31,
        "width": 1236,
        "height": 1036
      }
    }
  ],
  "refined_by": "rule",
  "image_captions": {
    "079d78bd9a2808bb_5_3": "这张图片展示了一个前馈神经网络语言模型（通常指NNLM，Neural Probabilistic Language Model）的架构示意图。该模型用于根据前文上下文预测下一个词的概率。以下是详细描述：\n\n### 整体结构\n图示是一个从底部输入到顶部输出的前向计算流程图。它描绘了一个具有单个隐藏层的神经网络。\n\n### 底部：输入层\n*   **输入**：模型的输入是词汇表中的词索引（index）。图中显示了三个索引：`index for w_{t-n+1}`， `index for w_{t-2}`， 和 `index for w_{t-1}`。这些代表了当前预测位置 `t` 之前的 `n-1` 个历史词。\n*   **词嵌入（Embedding）**：每个词索引通过一个共享的查找表（`Table look-up in C`）或矩阵（`Matrix C`）映射到一个连续的、低维的词向量表示。图中用 `C(w_{t-n+1})`， `C(w_{t-2})`， `C(w_{t-1})` 表示这些向量。\n*   **关键标注**：`shared parameters across words` 表明矩阵C中的参数（即每个词的向量表示）在所有词位置上是共享的。\n\n### 中间层：隐藏层\n*   **拼接**：来自输入层的所有词向量 `C(...)` 被拼接（concatenate）成一个长的向量。\n*   **隐藏层计算**：这个拼接后的向量被输入到一个全连接的隐藏层中。\n*   **激活函数**：隐藏层使用 `tanh`（双曲正切）作为激活函数进行非线性变换。\n\n### 顶部：输出层\n*   **计算核心**：从隐藏层到输出层的连接上标注了 `most computation here`，表明模型的主要计算开销发生在这里。这是因为输出层的维度通常等于词汇表的大小（可能非常大），需要计算每个候选词的得分。\n*   **输出转换**：隐藏层的输出被转换成一个与词汇表大小相同的向量。\n*   **概率输出**：该向量通过 `softmax` 函数，将每个值转换为一个概率值。图中用一组红色圆点代表输出向量。\n*   **最终目标**：顶部的公式 `i-th output = P(w_t = i | context)` 明确定义了第 `i` 个输出代表在给定前文上下文（`context`）的条件下，下一个词 `w_t` 是词表中第 `i` 个词的条件概率。\n\n### 数据流总结\n**输入词索引** → **通过共享矩阵C查找词向量** → **拼接所有上下文词向量** → **输入带有tanh激活的隐藏层** → **全连接到输出层（计算密集）** → **通过softmax得到下一个词的概率分布**。\n\n这张图清晰地描述了经典的基于神经网络的分布式语言模型的前馈计算流程，强调了参数共享（矩阵C）、非线性隐藏层（tanh）和概率输出（softmax）等关键技术点。"
  },
  "enriched_by": "rule",
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf",
  "file_size": 1786838,
  "doc_hash": "079d78bd9a2808bb",
  "image_refs": [
    "079d78bd9a2808bb_5_3"
  ],
  "tags": [
    "gpt",
    "transformer",
    "index",
    "模型",
    "tanh",
    "softmax",
    "神经",
    "优点"
  ],
  "doc_type": "pdf",
  "title": "整体结构",
  "summary": "神经⽹络特点： 优点：利⽤神经⽹络去建模当前词出现的概率与其前 n-1 个词之间的约束关系，很显然这种⽅式相 ⽐ n-gram 具有更好的泛化能⼒，只要词表征⾜够好。从⽽很⼤程度地降低了数据稀疏带来的问 题。缺点：对⻓序列的建模能⼒有限，可能会出现⻓距离遗忘以及训练时的梯度消失等问题，构建的模 型难以进⾏稳定的⻓⽂本输出。",
  "chunk_index": 4
}
```

</details>

**文本内容** (2147 字符):

```
神经⽹络特点：

优点：利⽤神经⽹络去建模当前词出现的概率与其前 n-1 个词之间的约束关系，很显然这种⽅式相
⽐ n-gram 具有更好的泛化能⼒，只要词表征⾜够好。从⽽很⼤程度地降低了数据稀疏带来的问
题。

缺点：对⻓序列的建模能⼒有限，可能会出现⻓距离遗忘以及训练时的梯度消失等问题，构建的模

型难以进⾏稳定的⻓⽂本输出。

2.3 基于Transformer的预训练语⾔模型

Transformer模型由⼀些编码器和解码器层组成（⻅图），学习复杂语义信息的能⼒强，很多主流预训
练模型在提取特征时都会选择Transformer结构，并产⽣了⼀系列的基于Transformer的预训练模型，
包括GPT、BERT、T5等.这些模型能够从⼤量的通⽤⽂本数据中学习⼤量的语⾔表示，并将这些知识运
⽤到下游任务中，获得了较好的效果.

预训练语⾔模型的使⽤⽅式：

预训练：预训练指建⽴基本的模型，先在⼀些⽐较基础的数据集、语料库上进⾏训练，然后按照具

体任务训练，学习数据的普遍特征。

微调：微调指在具体的下游任务中使⽤预训练好的模型进⾏迁移学习，以获取更好的泛化效果。

预训练语⾔模型的特点：

优点：更强⼤的泛化能⼒，丰富的语义表示，可以有效防⽌过拟合。

缺点：计算资源需求⼤，可解释性差等

2.4 ⼤语⾔模型

随着对预训练语⾔模型研究的开展，⼈们逐渐发现可能存在⼀种标度定律（Scaling Law），即随着预训
练模型参数的指数级提升，其语⾔模型性能也会线性上升。2020年，OpenAI发布了参数量⾼达1750亿
的GPT-3，⾸次展示了⼤语⾔模型的性能。

这张图片展示了一个前馈神经网络语言模型（通常指NNLM，Neural Probabilistic Language Model）的架构示意图。该模型用于根据前文上下文预测下一个词的概率。以下是详细描述：

### 整体结构
图示是一个从底部输入到顶部输出的前向计算流程图。它描绘了一个具有单个隐藏层的神经网络。

### 底部：输入层
*   **输入**：模型的输入是词汇表中的词索引（index）。图中显示了三个索引：`index for w_{t-n+1}`， `index for w_{t-2}`， 和 `index for w_{t-1}`。这些代表了当前预测位置 `t` 之前的 `n-1` 个历史词。
*   **词嵌入（Embedding）**：每个词索引通过一个共享的查找表（`Table look-up in C`）或矩阵（`Matrix C`）映射到一个连续的、低维的词向量表示。图中用 `C(w_{t-n+1})`， `C(w_{t-2})`， `C(w_{t-1})` 表示这些向量。
*   **关键标注**：`shared parameters across words` 表明矩阵C中的参数（即每个词的向量表示）在所有词位置上是共享的。

### 中间层：隐藏层
*   **拼接**：来自输入层的所有词向量 `C(...)` 被拼接（concatenate）成一个长的向量。
*   **隐藏层计算**：这个拼接后的向量被输入到一个全连接的隐藏层中。
*   **激活函数**：隐藏层使用 `tanh`（双曲正切）作为激活函数进行非线性变换。

### 顶部：输出层
*   **计算核心**：从隐藏层到输出层的连接上标注了 `most computation here`，表明模型的主要计算开销发生在这里。这是因为输出层的维度通常等于词汇表的大小（可能非常大），需要计算每个候选词的得分。
*   **输出转换**：隐藏层的输出被转换成一个与词汇表大小相同的向量。
*   **概率输出**：该向量通过 `softmax` 函数，将每个值转换为一个概率值。图中用一组红色圆点代表输出向量。
*   **最终目标**：顶部的公式 `i-th output = P(w_t = i | context)` 明确定义了第 `i` 个输出代表在给定前文上下文（`context`）的条件下，下一个词 `w_t` 是词表中第 `i` 个词的条件概率。

### 数据流总结
**输入词索引** → **通过共享矩阵C查找词向量** → **拼接所有上下文词向量** → **输入带有tanh激活的隐藏层** → **全连接到输出层（计算密集）** → **通过softmax得到下一个词的概率分布**。

这张图清晰地描述了经典的基于神经网络的分布式语言模型的前馈计算流程，强调了参数共享（矩阵C）、非线性隐藏层（tanh）和概率输出（softmax）等关键技术点。

相较于此前的参数量较⼩的预训练语⾔模型，例如，3.3亿参数的Bert-large和17亿参数的GPT-2，GPT-
3展现了在Few-shot语⾔任务能⼒上的⻜跃，并具备了预训练语⾔模型不具备的⼀些能⼒。后续将这种
现象称为能⼒涌现。例如，GPT-3能进⾏上下⽂学习，在不调整权重的情况下仅依据⽤户给出的任务示
例完成后续任务。这种能⼒⽅⾯的⻜跃引发研究界在⼤语⾔模型上的研究热潮，各⼤科技巨头纷纷推出
```

---

### Chunk 6: `079d78bd9a2808bb_0005_d8a228a3`

| 字段 | 值 |
|------|-----|
| chunk_index | 5 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | 1. 输入与编码器（左侧，负责理解输入序列语义） |
| refined_by | rule |
| enriched_by | rule |

**tags**: bleu, gram, add, norm, nice, day, today, 预测

**summary**: 参数量巨⼤的语⾔模型，例如，Meta公司1300亿参数量的LLaMA模型以及⾕歌公司5400亿参数量的 PaLM。国内如百度推出的⽂⼼⼀⾔ERNIE系列、清华⼤学团队推出的GLM系列，等等。⼤语⾔模型的特点： 优点：像“⼈类”⼀样智能，具备了能与⼈类沟通聊天的能⼒，甚⾄具备了使⽤插件进⾏⾃动信息检 索的能⼒ 缺点：参数量⼤，算⼒要求⾼、⽣成部分有害的、有偏⻅的内容等等 3 语⾔模型的评估指标 3.

**图片引用** (1 张):

| image_id | page | 尺寸 | text_offset | 文件路径 |
|----------|------|------|-------------|----------|
| `079d78bd9a2808bb_6_4` | p6 | 463×663 | 4431 | `079d78bd9a2808bb_6_4.png` |

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "title": "1. 输入与编码器（左侧，负责理解输入序列语义）",
  "image_refs": [
    "079d78bd9a2808bb_6_4"
  ],
  "refined_by": "rule",
  "image_captions": {
    "079d78bd9a2808bb_6_4": "这是**经典Transformer序列到序列（Seq2Seq）模型的架构图**，是自然语言处理领域的里程碑架构，用于机器翻译、文本生成等任务，整体分为左侧编码器、右侧解码器，最终输出序列的概率分布，以下是分层技术细节：\n\n### 1. 输入与编码器（左侧，负责理解输入序列语义）\n- 输入起点：`Inputs`（输入文本序列）→ 进入`Input Embedding`（输入嵌入层），将离散的文本token转为连续向量表示。\n- 位置信息注入：嵌入向量与`Positional Encoding`（位置编码，⊕表示相加操作）结合，为Transformer注入序列位置信息——因为Transformer本身没有循环/卷积结构，无法捕捉序列顺序，需要显式位置编码。\n- 重复堆叠的**Nx层编码器模块**（`Nx`表示重复N次，N通常为6/12）：\n  1.  `Multi-Head Attention`（多头自注意力机制）：并行多个注意力头，捕捉输入序列中任意位置的语义关联，让模型学习序列内的依赖关系。\n  2.  `Add & Norm`（残差连接+层归一化）：对注意力输出做残差（与输入向量相加），再执行层归一化，缓解深层网络的梯度消失问题，稳定训练过程。\n  3.  `Feed Forward`（前馈神经网络）：全连接网络，对注意力输出做非线性特征变换。\n  4.  `Add & Norm`：再次对前馈输出执行残差+归一化，完成一个编码器块的计算。\n\n### 2. 输出与解码器（右侧，负责生成目标序列）\n- 输出起点：`Outputs`（训练时为目标序列的偏移版本，推理时为上一步生成的输出）→ 进入`Output Embedding`（输出嵌入层），转为向量表示，同样加入`Positional Encoding`注入位置信息。\n- 重复堆叠的**Nx层解码器模块**：\n  1.  `Masked Multi-Head Attention`（带掩码的多头自注意力）：通过掩码（Mask）屏蔽未来位置的token，保证自回归生成时，模型只能关注当前位置及之前的序列信息，避免训练时“泄露”未来信息。\n  2.  `Add & Norm`：残差连接+层归一化，逻辑与编码器一致。\n  3.  `Multi-Head Attention`（交叉多头注意力）：输入包含解码器自身的注意力输出+编码器的最终输出，让解码器在生成时关联输入序列的语义，实现“输入-输出”的语义对齐。\n  4.  `Add & Norm`：残差连接+层归一化。\n  5.  `Feed Forward`（前馈神经网络）：全连接非线性变换。\n  6.  `Add & Norm`：残差+归一化，完成解码器块的计算。\n\n### 3. 最终输出层\n解码器的最终输出 → `Linear`（线性层，全连接网络，将解码器输出映射到词汇表维度）→ `Softmax`（Softmax层，将线性输出转为概率分布）→ 最终得到`Output Probabilities`（输出概率分布，对应词汇表中每个token的概率，取最大概率的token作为预测输出）。\n\n该架构的核心技术价值：摆脱了传统RNN/CNN的序列依赖，通过多头注意力机制实现长距离序列依赖建模，残差+归一化解决深层网络训练问题，位置编码补充序列顺序信息，是现代大语言模型的基础架构。"
  },
  "chunk_index": 5,
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf",
  "doc_type": "pdf",
  "doc_hash": "079d78bd9a2808bb",
  "images": [
    {
      "id": "079d78bd9a2808bb_6_4",
      "path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\images\\default\\079d78bd9a2808bb\\079d78bd9a2808bb_6_4.png",
      "page": 6,
      "text_offset": 4431,
      "text_length": 29,
      "position": {
        "xref": 35,
        "width": 463,
        "height": 663
      }
    }
  ],
  "summary": "参数量巨⼤的语⾔模型，例如，Meta公司1300亿参数量的LLaMA模型以及⾕歌公司5400亿参数量的 PaLM。国内如百度推出的⽂⼼⼀⾔ERNIE系列、清华⼤学团队推出的GLM系列，等等。⼤语⾔模型的特点： 优点：像“⼈类”⼀样智能，具备了能与⼈类沟通聊天的能⼒，甚⾄具备了使⽤插件进⾏⾃动信息检 索的能⼒ 缺点：参数量⼤，算⼒要求⾼、⽣成部分有害的、有偏⻅的内容等等 3 语⾔模型的评估指标 3.",
  "file_size": 1786838,
  "enriched_by": "rule",
  "tags": [
    "bleu",
    "gram",
    "add",
    "norm",
    "nice",
    "day",
    "today",
    "预测"
  ]
}
```

</details>

**文本内容** (2247 字符):

```
参数量巨⼤的语⾔模型，例如，Meta公司1300亿参数量的LLaMA模型以及⾕歌公司5400亿参数量的
PaLM。国内如百度推出的⽂⼼⼀⾔ERNIE系列、清华⼤学团队推出的GLM系列，等等。

⼤语⾔模型的特点：

优点：像“⼈类”⼀样智能，具备了能与⼈类沟通聊天的能⼒，甚⾄具备了使⽤插件进⾏⾃动信息检
索的能⼒

缺点：参数量⼤，算⼒要求⾼、⽣成部分有害的、有偏⻅的内容等等

3 语⾔模型的评估指标

3.1 BLEU

BLEU （双语评估替补）分数是评估⼀种语⾔翻译成另⼀种语⾔的⽂本质量的指标。它将“质量”的好坏
定义为与⼈类翻译结果的⼀致性程度。

BLEU算法实际上就是在判断两个句⼦的相似程度. BLEU 的分数取值范围是 0～1，分数越接近1，说明
翻译的质量越⾼。

BLEU有许多变种，根据 n-gram 可以划分成多种评价指标，常⻅的评价指标有BLEU-1、BLEU-2、

BLEU-3、BLEU-4四种，其中 n-gram 指的是连续的单词个数为n，BLEU-1衡量的是单词级别的准确

性，更⾼阶的BLEU可以衡量句⼦的流畅性.实践中，通常是取N=1~4，然后对进⾏加权平均。

下⾯举例说计算过程：

基本步骤：

分别计算预测⽂本和⽬标⽂本的N-grams模型，然后统计其匹配的个数，计算匹配度:
公式：预测⽂本中正确预测的 n−gram 的个数 /预测⽂本中所有n−gram 的个数

这是**经典Transformer序列到序列（Seq2Seq）模型的架构图**，是自然语言处理领域的里程碑架构，用于机器翻译、文本生成等任务，整体分为左侧编码器、右侧解码器，最终输出序列的概率分布，以下是分层技术细节：

### 1. 输入与编码器（左侧，负责理解输入序列语义）
- 输入起点：`Inputs`（输入文本序列）→ 进入`Input Embedding`（输入嵌入层），将离散的文本token转为连续向量表示。
- 位置信息注入：嵌入向量与`Positional Encoding`（位置编码，⊕表示相加操作）结合，为Transformer注入序列位置信息——因为Transformer本身没有循环/卷积结构，无法捕捉序列顺序，需要显式位置编码。
- 重复堆叠的**Nx层编码器模块**（`Nx`表示重复N次，N通常为6/12）：
  1.  `Multi-Head Attention`（多头自注意力机制）：并行多个注意力头，捕捉输入序列中任意位置的语义关联，让模型学习序列内的依赖关系。
  2.  `Add & Norm`（残差连接+层归一化）：对注意力输出做残差（与输入向量相加），再执行层归一化，缓解深层网络的梯度消失问题，稳定训练过程。
  3.  `Feed Forward`（前馈神经网络）：全连接网络，对注意力输出做非线性特征变换。
  4.  `Add & Norm`：再次对前馈输出执行残差+归一化，完成一个编码器块的计算。

### 2. 输出与解码器（右侧，负责生成目标序列）
- 输出起点：`Outputs`（训练时为目标序列的偏移版本，推理时为上一步生成的输出）→ 进入`Output Embedding`（输出嵌入层），转为向量表示，同样加入`Positional Encoding`注入位置信息。
- 重复堆叠的**Nx层解码器模块**：
  1.  `Masked Multi-Head Attention`（带掩码的多头自注意力）：通过掩码（Mask）屏蔽未来位置的token，保证自回归生成时，模型只能关注当前位置及之前的序列信息，避免训练时“泄露”未来信息。
  2.  `Add & Norm`：残差连接+层归一化，逻辑与编码器一致。
  3.  `Multi-Head Attention`（交叉多头注意力）：输入包含解码器自身的注意力输出+编码器的最终输出，让解码器在生成时关联输入序列的语义，实现“输入-输出”的语义对齐。
  4.  `Add & Norm`：残差连接+层归一化。
  5.  `Feed Forward`（前馈神经网络）：全连接非线性变换。
  6.  `Add & Norm`：残差+归一化，完成解码器块的计算。

### 3. 最终输出层
解码器的最终输出 → `Linear`（线性层，全连接网络，将解码器输出映射到词汇表维度）→ `Softmax`（Softmax层，将线性输出转为概率分布）→ 最终得到`Output Probabilities`（输出概率分布，对应词汇表中每个token的概率，取最大概率的token作为预测输出）。

该架构的核心技术价值：摆脱了传统RNN/CNN的序列依赖，通过多头注意力机制实现长距离序列依赖建模，残差+归一化解决深层网络训练问题，位置编码补充序列顺序信息，是现代大语言模型的基础架构。

假设分别给出⼀个预测⽂本和⽬标⽂本如下：

预测⽂本: It is a nice day today

⽬标⽂本: today is a nice day

使⽤1-gram进⾏匹配

预测⽂本: {it, is, a, nice, day, today}

⽬标⽂本: {today, is, a, nice, day}

结果:

 其中{today, is, a, nice, day}匹配，所以匹配度为5/6

使⽤2-gram进⾏匹配
```

---

### Chunk 7: `079d78bd9a2808bb_0006_defc85ab`

| 字段 | 值 |
|------|-----|
| chunk_index | 6 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | ⽬标⽂本: today is a nice day |
| refined_by | rule |
| enriched_by | rule |

**tags**: nice, day, today, 匹配, 预测, gram, 结果, 所以匹配度为

**summary**: ⽬标⽂本: today is a nice day 使⽤1-gram进⾏匹配 预测⽂本: {it, is, a, nice, day, today} ⽬标⽂本: {today, is, a, nice

**图片**: 无

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "title": "⽬标⽂本: today is a nice day",
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf",
  "enriched_by": "rule",
  "doc_type": "pdf",
  "doc_hash": "079d78bd9a2808bb",
  "tags": [
    "nice",
    "day",
    "today",
    "匹配",
    "预测",
    "gram",
    "结果",
    "所以匹配度为"
  ],
  "file_size": 1786838,
  "summary": "⽬标⽂本: today is a nice day 使⽤1-gram进⾏匹配 预测⽂本: {it, is, a, nice, day, today} ⽬标⽂本: {today, is, a, nice",
  "chunk_index": 6,
  "refined_by": "rule"
}
```

</details>

**文本内容** (872 字符):

```
⽬标⽂本: today is a nice day

使⽤1-gram进⾏匹配

预测⽂本: {it, is, a, nice, day, today}

⽬标⽂本: {today, is, a, nice, day}

结果:

 其中{today, is, a, nice, day}匹配，所以匹配度为5/6

使⽤2-gram进⾏匹配

预测⽂本: {it is, is a, a nice, nice day, day today}

⽬标⽂本: {today is, is a, a nice, nice day}

结果:

 其中{is a, a nice, nice day}匹配，所以匹配度为3/5

使⽤3-gram进⾏匹配

预测⽂本: {it is a, is a nice, a nice day, nice day today}

⽬标⽂本: {today is a, is a nice, a nice day}

结果:

 其中{is a nice, a nice day}匹配，所以匹配度为2/4

使⽤4-gram进⾏匹配

预测⽂本: {it is a nice, is a nice day, a nice day today}

⽬标⽂本: {today is a nice, is a nice day}

结果:

 其中{is a nice day}匹配，所以匹配度为1/3

上述例⼦会出现⼀种极端情况，请看下⾯示例：

预测⽂本: the the the the

⽬标⽂本: The cat is standing on the ground

如果按照1-gram的⽅法进⾏匹配，则匹配度为1，显然是不合理的，所以计算某个词的出现次数进⾏改进

将计算某个词正确预测次数的⽅法改为计算某个词在⽂本中出现的最⼩次数,如下所示的公式：

其中 表示在预测⽂本中出现的第 个词语, 则代表在预测⽂本中这个词语出现的次数，⽽ 则代
表在⽬标⽂本中这个词语出现的次数。

python代码实现：
```

---

### Chunk 8: `079d78bd9a2808bb_0007_87281c51`

| 字段 | 值 |
|------|-----|
| chunk_index | 7 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | 第⼀步安装nltk的包-->pip install nltk |
| refined_by | rule |
| enriched_by | rule |

**tags**: bleu, gram, sentence, reference, candidate, weights, print, nltk

**summary**: 如果按照1-gram的⽅法进⾏匹配，则匹配度为1，显然是不合理的，所以计算某个词的出现次数进⾏改进 将计算某个词正确预测次数的⽅法改为计算某个词在⽂本中出现的最⼩次数,如下所示的公式： 其中 表示在预测⽂本中出现的第 个词语, 则代表在预测⽂本中这个词语出现的次数，⽽ 则代 表在⽬标⽂本中这个词语出现的次数。python代码实现： from nltk.translate.

**图片**: 无

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "doc_hash": "079d78bd9a2808bb",
  "chunk_index": 7,
  "enriched_by": "rule",
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf",
  "title": "第⼀步安装nltk的包-->pip install nltk",
  "tags": [
    "bleu",
    "gram",
    "sentence",
    "reference",
    "candidate",
    "weights",
    "print",
    "nltk"
  ],
  "doc_type": "pdf",
  "file_size": 1786838,
  "refined_by": "rule",
  "summary": "如果按照1-gram的⽅法进⾏匹配，则匹配度为1，显然是不合理的，所以计算某个词的出现次数进⾏改进 将计算某个词正确预测次数的⽅法改为计算某个词在⽂本中出现的最⼩次数,如下所示的公式： 其中 表示在预测⽂本中出现的第 个词语, 则代表在预测⽂本中这个词语出现的次数，⽽ 则代 表在⽬标⽂本中这个词语出现的次数。python代码实现： from nltk.translate."
}
```

</details>

**文本内容** (866 字符):

```
如果按照1-gram的⽅法进⾏匹配，则匹配度为1，显然是不合理的，所以计算某个词的出现次数进⾏改进

将计算某个词正确预测次数的⽅法改为计算某个词在⽂本中出现的最⼩次数,如下所示的公式：

其中 表示在预测⽂本中出现的第 个词语, 则代表在预测⽂本中这个词语出现的次数，⽽ 则代
表在⽬标⽂本中这个词语出现的次数。

python代码实现：

# 第⼀步安装nltk的包-->pip install nltk

from nltk.translate.bleu_score import sentence_bleu

def cumulative_bleu(reference, candidate):

 bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))

 bleu_2_gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0,

0))

 bleu_3_gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33,

0.33, 0))

 bleu_4_gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25,

0.25, 0.25))

 # print('bleu 1-gram: %f' % bleu_1_gram)

 # print('bleu 2-gram: %f' % bleu_2_gram)

 # print('bleu 3-gram: %f' % bleu_3_gram)

 # print('bleu 4-gram: %f' % bleu_4_gram)

 return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram

# 预测⽂本
```

---

### Chunk 9: `079d78bd9a2808bb_0008_26cd02b3`

| 字段 | 值 |
|------|-----|
| chunk_index | 8 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | print('bleu 3-gram: %f' % bleu_3_gram) |
| refined_by | rule |
| enriched_by | rule |

**tags**: bleu, rouge, gram, text, reference, print, 指标, candidate

**summary**: # print('bleu 4-gram: %f' % bleu4gram) return bleu1gram, bleu2gram, bleu3gram, bleu4gram candidatete

**图片**: 无

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "chunk_index": 8,
  "tags": [
    "bleu",
    "rouge",
    "gram",
    "text",
    "reference",
    "print",
    "指标",
    "candidate"
  ],
  "enriched_by": "rule",
  "file_size": 1786838,
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf",
  "doc_hash": "079d78bd9a2808bb",
  "doc_type": "pdf",
  "title": "print('bleu 3-gram: %f' % bleu_3_gram)",
  "summary": "# print('bleu 4-gram: %f' % bleu4gram) return bleu1gram, bleu2gram, bleu3gram, bleu4gram candidatete",
  "refined_by": "rule"
}
```

</details>

**文本内容** (849 字符):

```
# print('bleu 3-gram: %f' % bleu_3_gram)

 # print('bleu 4-gram: %f' % bleu_4_gram)

 return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram

# 预测⽂本

candidate_text = ["This", "is", "some", "generated", "text"]

# ⽬标⽂本列表

reference_texts = [["This", "is", "a", "reference", "text"],

 ["This", "is", "another", "reference", "text"]]

# 计算 Bleu 指标

c_bleu = cumulative_bleu(reference_texts, candidate_text)

# 打印结果

print("The Bleu score is:", c_bleu)

# The Bleu score is: (0.6, 0.387, 1.5949011744633917e-102, 9.283142785759642e-

155)

3.2 ROUGE

ROUGE指标是在机器翻译、⾃动摘要、问答⽣成等领域常⻅的评估指标。ROUGE通过将模型⽣成的摘
要或者回答与参考答案（⼀般是⼈⼯⽣成的）进⾏⽐较计算，得到对应的得分。

ROUGE指标与BLEU指标⾮常类似，均可⽤来衡量⽣成结果和标准结果的匹配程度，不同的是ROUGE基
于召回率，BLEU更看重准确率。

ROUGE分为四种⽅法：ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S.

下⾯举例说计算过程（这⾥只介绍ROUGE_N）：

基本步骤：

Rouge-N实际上是将模型⽣成的结果和标准结果按N-gram拆分后，计算召回率

假设模型预测⽂本和⼀个⽬标⽂本如下：
```

---

### Chunk 10: `079d78bd9a2808bb_0009_d495574c`

| 字段 | 值 |
|------|-----|
| chunk_index | 9 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | 第⼀步：安装rouge-->pip install rouge |
| refined_by | rule |
| enriched_by | rule |

**tags**: rouge, nice, day, today, text, generated, reference, scores

**summary**: ROUGE分为四种⽅法：ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S.

**图片**: 无

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "refined_by": "rule",
  "doc_type": "pdf",
  "title": "第⼀步：安装rouge-->pip install rouge",
  "file_size": 1786838,
  "summary": "ROUGE分为四种⽅法：ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S.",
  "chunk_index": 9,
  "enriched_by": "rule",
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf",
  "doc_hash": "079d78bd9a2808bb",
  "tags": [
    "rouge",
    "nice",
    "day",
    "today",
    "text",
    "generated",
    "reference",
    "scores"
  ]
}
```

</details>

**文本内容** (885 字符):

```
ROUGE分为四种⽅法：ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S.

下⾯举例说计算过程（这⾥只介绍ROUGE_N）：

基本步骤：

Rouge-N实际上是将模型⽣成的结果和标准结果按N-gram拆分后，计算召回率

假设模型预测⽂本和⼀个⽬标⽂本如下：

预测⽂本: It is a nice day today

⽬标⽂本: Today is a nice day

使⽤ROUGE-1进⾏匹配

预测⽂本: {it, is, a, nice, day, today}

⽬标⽂本: {today, is, a, nice, day}

结果:

 :其中{today, is, a, nice, day}匹配，所以匹配度为5/5=1,这说明⽣成的内容完全覆盖了参考

⽂本中的所有单词，质量较⾼。

通过类似的⽅法，可以计算出其他ROUGE指标（如ROUGE-2、ROUGE-L、ROUGE-S）的评分，
从⽽综合评估系统⽣成的⽂本质量。

python代码实现：

# 第⼀步：安装rouge-->pip install rouge

from rouge import Rouge

# 预测⽂本

generated_text = "This is some generated text."

# ⽬标⽂本列表

reference_texts = ["This is a reference text.", "This is another generated

reference text."]

# 计算 ROUGE 指标

rouge = Rouge()

scores = rouge.get_scores(generated_text, reference_texts[1])

# 打印结果

print("ROUGE-1 precision:", scores[0]["rouge-1"]["p"])

print("ROUGE-1 recall:", scores[0]["rouge-1"]["r"])
```

---

### Chunk 11: `079d78bd9a2808bb_0010_051f6cc2`

| 字段 | 值 |
|------|-----|
| chunk_index | 10 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | 打印结果 |
| refined_by | rule |
| enriched_by | rule |

**tags**: rouge, scores, print, ppl, precision, recall, score, perplexity

**summary**: scores = rouge.

**图片**: 无

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "file_size": 1786838,
  "summary": "scores = rouge.",
  "enriched_by": "rule",
  "doc_type": "pdf",
  "title": "打印结果",
  "refined_by": "rule",
  "chunk_index": 10,
  "doc_hash": "079d78bd9a2808bb",
  "tags": [
    "rouge",
    "scores",
    "print",
    "ppl",
    "precision",
    "recall",
    "score",
    "perplexity"
  ],
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf"
}
```

</details>

**文本内容** (855 字符):

```
scores = rouge.get_scores(generated_text, reference_texts[1])

# 打印结果

print("ROUGE-1 precision:", scores[0]["rouge-1"]["p"])

print("ROUGE-1 recall:", scores[0]["rouge-1"]["r"])

print("ROUGE-1 F1 score:", scores[0]["rouge-1"]["f"])

# ROUGE-1 precision: 0.8

# ROUGE-1 recall: 0.6666666666666666

# ROUGE-1 F1 score: 0.7272727223140496

3.3 困惑度PPL(perplexity)

PPL⽤来度量⼀个概率分布或概率模型预测样本的好坏程度。

PPL基本思想:

给测试集的句⼦赋予较⾼概率值的语⾔模型较好,当语⾔模型训练完之后，测试集中的句⼦都是正
常的句⼦，那么训练好的模型就是在测试集上的概率越⾼越好.
基本公式（两种⽅式）：

由公式可知，句⼦概率越⼤，语⾔模型越好，迷惑度越⼩。

python代码实现：

import math

# 定义语料库

sentences = [

['I', 'have', 'a', 'pen'],

['He', 'has', 'a', 'book'],

['She', 'has', 'a', 'cat']

]

# 定义语⾔模型

unigram = {

'I': 1/12,

'have': 1/12,

'a': 3/12,

'pen': 1/12,

'He': 1/12,

'has': 2/12,

'book': 1/12,

'She': 1/12,

'cat': 1/12

}

# 计算困惑度

perplexity = 0

for sentence in sentences:
```

---

### Chunk 12: `079d78bd9a2808bb_0011_3d9eb1ff`

| 字段 | 值 |
|------|-----|
| chunk_index | 11 |
| source_path | `D:\AI\my_AI_project\MODULAR-RAG-MCP-SERVER\data\documents\LLM基础知识.pdf` |
| doc_hash | `079d78bd9a2808bb` |
| doc_type | pdf |
| file_size | 1786838 字节 |
| title | 计算困惑度 |
| refined_by | rule |
| enriched_by | rule |

**tags**: perplexity, sentence, log, frac, prob, 困惑度, 为底, 困惑度为

**summary**: 'a': 3/12, 'pen': 1/12, 'He': 1/12, 'has': 2/12, 'book': 1/12, 'She': 1/12, 'cat': 1/12 } perplexity

**图片引用** (2 张):

| image_id | page | 尺寸 | text_offset | 文件路径 |
|----------|------|------|-------------|----------|
| `079d78bd9a2808bb_11_5` | p11 | 471×65 | 8258 | `079d78bd9a2808bb_11_5.png` |
| `079d78bd9a2808bb_11_6` | p11 | 249×52 | 8290 | `079d78bd9a2808bb_11_6.png` |

<details>
<summary>完整元数据 JSON</summary>

```json
{
  "image_captions": {
    "079d78bd9a2808bb_11_5": "这张图片展示了自然语言处理（NLP）领域中**困惑度（Perplexity，符号为$PP(W)$）**的数学定义公式：\n$$PP(W)=P(w_1w_2...w_N)^{-\\frac{1}{N}}=\\sqrt[N]{\\frac{1}{P(w_1w_2...w_N)}}$$\n### 公式细节与含义：\n1.  元素定义：\n    - $W$：指代目标文本序列，由$w_1、w_2...w_N$共N个词元（token）构成；\n    - $P(w_1w_2...w_N)$：是该文本序列的联合概率，代表该序列在对应语言模型中的生成/出现概率；\n    - $N$：是文本序列的词元长度。\n2.  核心定义：\n    困惑度的计算等价于该序列联合概率的$-1/N$次方，也可通过“$N$次根号下（$1$除以序列联合概率）”推导。\n3.  应用价值：\n    它是衡量语言模型性能的核心指标，困惑度数值越低，说明语言模型对文本的预测建模能力越强，模型表现越优秀。",
    "079d78bd9a2808bb_11_6": "这是一张展示自然语言处理领域核心指标——**困惑度，Perplexity，简称PP**计算公式的图片，具体公式及元素含义如下：\n\n### 公式内容\n$$PP(S) = 2^{-\\frac{1}{N} \\sum \\log(P(w_i))}$$\n\n### 符号与技术含义\n1.  $PP(S)$：代表句子$S$的困惑度，是衡量语言模型对文本预测能力的关键指标；\n2.  $N$：为句子$S$的总词长（句中包含的词的总数）；\n3.  $\\sum \\log(P(w_i))$：对句子中每个词$w_i$的对数概率做求和操作：\n    - $P(w_i)$：指语言模型在给定上下文时，预测词$w_i$出现的概率；\n    - 此处的$\\log$默认以2为底，让困惑度的单位为比特，更贴合信息度量的逻辑；\n4.  整体逻辑：通过归一化词概率的对数和，再以2为底求指数，得到句子的困惑度。该指标数值越小，说明模型对句子的预测不确定性越低，模型的语言预测性能越好。"
  },
  "refined_by": "rule",
  "enriched_by": "rule",
  "summary": "'a': 3/12, 'pen': 1/12, 'He': 1/12, 'has': 2/12, 'book': 1/12, 'She': 1/12, 'cat': 1/12 } perplexity",
  "chunk_index": 11,
  "title": "计算困惑度",
  "tags": [
    "perplexity",
    "sentence",
    "log",
    "frac",
    "prob",
    "困惑度",
    "为底",
    "困惑度为"
  ],
  "file_size": 1786838,
  "doc_type": "pdf",
  "image_refs": [
    "079d78bd9a2808bb_11_5",
    "079d78bd9a2808bb_11_6"
  ],
  "doc_hash": "079d78bd9a2808bb",
  "images": [
    {
      "id": "079d78bd9a2808bb_11_5",
      "path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\images\\default\\079d78bd9a2808bb\\079d78bd9a2808bb_11_5.png",
      "page": 11,
      "text_offset": 8258,
      "text_length": 30,
      "position": {
        "xref": 53,
        "width": 471,
        "height": 65
      }
    },
    {
      "id": "079d78bd9a2808bb_11_6",
      "path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\images\\default\\079d78bd9a2808bb\\079d78bd9a2808bb_11_6.png",
      "page": 11,
      "text_offset": 8290,
      "text_length": 30,
      "position": {
        "xref": 54,
        "width": 249,
        "height": 52
      }
    }
  ],
  "source_path": "D:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER\\data\\documents\\LLM基础知识.pdf"
}
```

</details>

**文本内容** (1352 字符):

```
'a': 3/12,

'pen': 1/12,

'He': 1/12,

'has': 2/12,

'book': 1/12,

'She': 1/12,

'cat': 1/12

}

# 计算困惑度

perplexity = 0

for sentence in sentences:

这张图片展示了自然语言处理（NLP）领域中**困惑度（Perplexity，符号为$PP(W)$）**的数学定义公式：
$$PP(W)=P(w_1w_2...w_N)^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_1w_2...w_N)}}$$
### 公式细节与含义：
1.  元素定义：
    - $W$：指代目标文本序列，由$w_1、w_2...w_N$共N个词元（token）构成；
    - $P(w_1w_2...w_N)$：是该文本序列的联合概率，代表该序列在对应语言模型中的生成/出现概率；
    - $N$：是文本序列的词元长度。
2.  核心定义：
    困惑度的计算等价于该序列联合概率的$-1/N$次方，也可通过“$N$次根号下（$1$除以序列联合概率）”推导。
3.  应用价值：
    它是衡量语言模型性能的核心指标，困惑度数值越低，说明语言模型对文本的预测建模能力越强，模型表现越优秀。

这是一张展示自然语言处理领域核心指标——**困惑度，Perplexity，简称PP**计算公式的图片，具体公式及元素含义如下：

### 公式内容
$$PP(S) = 2^{-\frac{1}{N} \sum \log(P(w_i))}$$

### 符号与技术含义
1.  $PP(S)$：代表句子$S$的困惑度，是衡量语言模型对文本预测能力的关键指标；
2.  $N$：为句子$S$的总词长（句中包含的词的总数）；
3.  $\sum \log(P(w_i))$：对句子中每个词$w_i$的对数概率做求和操作：
    - $P(w_i)$：指语言模型在给定上下文时，预测词$w_i$出现的概率；
    - 此处的$\log$默认以2为底，让困惑度的单位为比特，更贴合信息度量的逻辑；
4.  整体逻辑：通过归一化词概率的对数和，再以2为底求指数，得到句子的困惑度。该指标数值越小，说明模型对句子的预测不确定性越低，模型的语言预测性能越好。

 sentence_prob = 1

 for word in sentence:

 sentence_prob *= unigram[word]

 sentence_perplexity = 1/sentence_prob

 perplexity += math.log(sentence_perplexity, 2) #以2为底

perplexity = 2 ** (-perplexity/len(sentences))

print('困惑度为：', perplexity)

# 困惑度为： 0.0002296

⼩结总结

本⼩节主要介绍LLM的背景知识，了解⽬前LLM发展基本历程
对语⾔模型的类别分别进⾏了介绍，如基于统计的N-gram模型，以及深度学习的神经⽹络模型
```

---

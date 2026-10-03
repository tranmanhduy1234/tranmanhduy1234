<div align="center">

# Hi, I'm Mạnh Duy 👋

### AI Researcher / Research Engineer in progress

**Vision · Language · Multimodal Learning · Physical AI**

I like building AI systems far enough below the API layer that I can understand  
**what the model sees, how it learns, why it fails, and what to try next.**

</div>

---

## 🧭 A little about me

I started from **Software Engineering**, then gradually moved deeper into AI by building systems across different modalities.

My path so far has been roughly:

```text
Machine Translation / ASR
        ↓
Language & Sequence Modeling

Object Detection / Landmark Detection
        ↓
Visual Representation & Perception

Language + Vision
        ↓
VLM → VLA → Physical AI
```

What keeps me interested is not simply making a model run. I enjoy the point where something does **not** work as expected, because that is usually where the real learning begins: tracing information flow, checking objectives, finding representation bottlenecks, debugging training dynamics, or discovering that the problem is actually in inference rather than the model itself.

---

## 🚧 What I'm working on now

### InA-Bridge — Vision–Language Model

My current main project is a VLM architecture built around the idea that **visual perception and language alignment do not have to be the same thing**.

```text
DINOv3
   ↓
Q-Former
   ↓
Projector
   ↓
Qwen3-4B
```

The idea is to preserve a rich self-supervised visual representation first, then learn how language should access that representation through a separate bridge.

Some of the parts I am implementing:

- 🧩 **Q-Former** with 128 learnable visual queries
- 🔀 custom cross-attention and objective-specific attention masks
- 🔗 **ITC / ITM / ITG** multimodal objectives
- 🧠 multi-positive contrastive learning
- 👨‍🏫 EMA momentum teacher + soft targets
- 🗃️ MoCo-style memory queue
- 🎯 similarity-based hard-negative mining
- ⚙️ mixed precision, gradient accumulation, clipping, warmup–cosine scheduling
- ☁️ FSDP / ZeRO-3 as the next scaling step

The project is still in progress. I am currently moving from local pipeline validation toward full DINOv3 integration and larger-scale distributed training.

The question I keep coming back to is:

> If vision is aligned too strongly with text from the beginning, what visual information do we lose simply because people rarely describe it in language?

That question is also one reason I am interested in **Physical AI**.

---

## 🧠 Projects I learned the most from

### 🌐 English → Vietnamese Machine Translation

This was one of the projects that taught me what an **end-to-end AI system** really means.

I worked through the entire pipeline:

```text
Raw corpora
→ cleaning
→ language / semantic filtering
→ tokenizer
→ Transformer
→ training
→ beam-search decoding
```

Highlights:

- 35M+ documented sentence pairs
- fastText language filtering + LaBSE semantic filtering
- 40K SentencePiece Unigram vocabulary
- 94.76M-parameter Transformer
- 6 encoder + 6 decoder layers
- pre-RMSNorm + SDPA attention
- teacher forcing, label smoothing, gradient accumulation
- custom batched beam search + KV cache
- 215K+ optimizer updates
- reported COMET: **0.725** on EVBCorpus 2.0

One useful failure I found was in cached decoding: KV caching is not enough by itself — positional state and beam-parent state also have to remain consistent. That pushed me to think much more carefully about **causality and inference state**, not just training loss.

---

### 👁️ YOLOv10-Style Object Detection

I implemented a YOLOv10-style detector directly in PyTorch because I wanted to understand what happens below `model.train()`.

Things I worked on:

- custom backbone blocks: C2f / CIB / SCDown / SPPF
- PAFPN multi-scale feature fusion
- three-scale decoupled detection heads
- one-to-many + one-to-one branches
- Task-Aligned Assignment
- BCE + CIoU + DFL objectives
- EMA, AMP, warmup–cosine scheduling
- Objects365-derived pretraining → COCO fine-tuning
- ONNX export and NMS-free runtime

A stored EMA evaluation reached **37.04% mAP50-95** on COCO validation.

The most interesting part was not the final number. Early training had relatively strong precision but weaker recall, and small-object / P3 behavior became a clear bottleneck. That forced me to look at **feature scale, assignment, object size, and augmentation** instead of treating mAP as one opaque score.

---

## 🎙️ Other things I've built

### Vietnamese ASR

- ~5,292 hours of audio
- ~1.35M processed samples
- 80-bin log-Mel spectrograms
- Conv1D frontend + Transformer encoder-decoder
- 145.2M parameters
- autoregressive decoding with cross-attention
- beam search + KV cache
- reported WER: **13.75%**

### Driver-state / Landmark Detection

I am also exploring fine-grained face / eye / mouth representations, temporal modeling, transfer learning, feature selection, metaheuristic optimization, and eventually reinforcement learning for adaptive driver-state systems.

---

## 🔬 How I like to work

My default loop is:

```text
Build → Understand → Diagnose → Experiment
```

**Build** — implement enough of the system to control the important mechanisms.  
**Understand** — follow tensors, gradients, objectives, representations, memory and compute.  
**Diagnose** — treat failure as information instead of hiding it.  
**Experiment** — change one meaningful thing and test a hypothesis.

I don't mind a project being unfinished if I can clearly explain:

- what currently works,
- what does not,
- why I think it fails,
- and what experiment should come next.

---

## 🌱 Where I want to go

Long term, I want to move toward:

**Multimodal Intelligence → VLA → Physical AI**

A robot operating in the real world cannot rely only on concepts that happen to be easy to describe with text. It needs geometry, spatial relationships, object state, temporal continuity, uncertainty, and persistent representations of the world.

Topics I am especially interested in:

- self-supervised vision
- multimodal representation learning
- contrastive learning
- spatial / temporal reasoning
- world models
- VLM / VLA
- reinforcement learning
- distributed training
- efficient multimodal systems

---

## 🛠️ Tech I use

`Python` `PyTorch` `Transformers` `CUDA` `Hugging Face`  
`OpenCV` `ONNX` `Docker` `Linux` `Git`  
`SentencePiece` `fastText` `LaBSE`  
`FSDP` `ZeRO-3` `AMP/BF16/FP16`

Foundations I care about: **linear algebra, probability, optimization, algorithms, information flow, and systems thinking.**

---

## 🇻🇳 Why this matters to me

Looking toward Vietnam's development goals for 2045, I believe a high-income economy cannot rely indefinitely on low-cost labor or remain concentrated at the end of global value chains.

That is one of the reasons I chose to take AI research seriously: I want to understand and help build core technology, not only consume it.

---

## 📫 Contact

- **Email:** trandomanhduy2004@gmail.com
- **Location:** Ho Chi Minh City, Vietnam
- **Education:** Posts and Telecommunications Institute of Technology (PTIT)

If you're working on a difficult AI problem involving **vision, language, multimodal learning, or physical intelligence**, I'd be interested in talking.

---

<div align="center">

### Build deeply. Understand mechanisms. Learn from failure.

</div>

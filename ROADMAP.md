# AI Learning Roadmap — RAG → Image → Video
## Complete Step-by-Step Guide

---

## ✅ COMPLETED (What you've already built)

- [x] Level 1: Basic Agent — `helloworld.py`
- [x] Level 2: Tool Calling — `tool_calling.py`
- [x] Level 3: Conversation Memory — `conversation_history.py`
- [x] Level 4: Context Providers — `agent_memory.py`
- [x] Level 5: Workflows — `workflow_demo.py`, `workflow_ai.py`

---

## PART 1: RAG & CHATBOTS (Month 1-2)

### Week 1-2: Foundations — Embeddings & Vector Search

**Goal:** Understand how AI finds "similar meaning"

**Build Project 1: Semantic Search Engine**
- Input: "How do I return a product?"
- Output: Finds "Refund policy: 30-day returns accepted..."

**Step 1: Text → Numbers (Embeddings)**
- Install sentence-transformers library
- Convert sentences to vectors (384 numbers each)
- Understand: WHY similar meanings = close vectors
- 🔨 Build: embed 50 sample sentences, print vectors

**Step 2: Compare Vectors (Similarity)**
- Cosine similarity (most common)
- Dot product
- Euclidean distance
- 🔨 Build: search function that finds top-5 similar sentences

**Step 3: Vector Database (ChromaDB)**
- Install ChromaDB (runs locally, free)
- Store 100+ documents
- Search with natural language queries
- 🔨 Build: document search engine over text files

**Step 4: Chunking Strategies**
- Fixed size chunks (500 chars)
- Sentence-based chunks
- Paragraph-based chunks
- Overlapping chunks (why overlap matters)
- 🔨 Build: compare search quality with each strategy

**Tools:** Python, sentence-transformers, ChromaDB, NumPy
**GPU Needed:** No

---

### Week 3-4: Naive RAG → Hybrid RAG

**Goal:** Build a working chatbot that answers from YOUR documents

**Build Project 2: Document Q&A Chatbot**

**Step 5: Naive RAG (simplest version)**
- Load documents (text, PDF)
- Chunk → embed → store in ChromaDB
- User asks question → embed → find similar chunks
- Feed top-5 chunks + question to AI → answer
- 🔨 Build: basic RAG pipeline end-to-end

**Step 6: Hybrid Search (Vector + Keyword)**
- Vector search: finds similar MEANING
- BM25 keyword search: finds exact WORDS
- Reciprocal Rank Fusion: merge both result lists
- 🔨 Build: compare naive vs hybrid search quality

**Step 7: Re-ranking**
- Cross-encoder re-ranker (more accurate than embeddings)
- Scores each (question, document) pair directly
- Slower but much more precise
- 🔨 Build: add re-ranking step, measure improvement

**Step 8: Source Citations**
- Track which document + page answered the question
- Show: "Based on report.pdf, page 3..."
- Highlight the exact passage used
- 🔨 Build: add citation tracking to chatbot

**Tools:** ChromaDB, PyMuPDF, rank_bm25, cross-encoder
**GPU Needed:** No for search, optional for cross-encoder

---

### Week 5-6: Advanced RAG Techniques

**Goal:** Make chatbot self-correcting and intelligent

**Step 9: Corrective RAG (CRAG)**
- After retrieval, AI CHECKS if documents are relevant
- If irrelevant → reformulate query → search again
- If partially relevant → extract only useful parts
- 🔨 Build: add relevance checker to pipeline

**Step 10: Self-RAG**
- AI generates answer + self-critique
- Check 1: Is the doc relevant to the question?
- Check 2: Is my answer supported by the doc?
- Check 3: Is my answer useful?
- If any check fails → retry with different approach
- 🔨 Build: add 3 self-checks after generation

**Step 11: Query Transformation**
- Query decomposition: complex → multiple simple queries
  - "Compare Q1 and Q2 revenue" → 3 sub-queries
- Query expansion: add related terms
  - "ML" → "ML OR machine learning OR deep learning"
- HyDE (Hypothetical Document Embeddings):
  - Generate fake answer first → use IT to search
- 🔨 Build: implement all 3 query transformation techniques

**Step 12: Contextual Retrieval (Anthropic's technique)**
- Before embedding, add context to each chunk
- Original: "Revenue grew 20%"
- Enhanced: "From Acme Corp 2025 report: Revenue grew 20%"
- AI adds context at indexing time
- 🔨 Build: re-index documents with contextual chunks

**Step 13: Agentic RAG**
- AI is the DRIVER — decides what to search & when
- Tools: search_docs, search_web, calculate, summarize
- AI decides: "I need more info" → searches again
- AI decides: "I have enough" → generates answer
- 🔨 Build: agent with search tools + reasoning loop

**Tools:** agent_framework, ChromaDB, sentence-transformers
**GPU Needed:** No (uses Azure OpenAI API)

---

### Week 7-8: Production Chatbot

**Goal:** Build a complete, polished chatbot

**Build Project 3: AI Knowledge Assistant (Full)**

**Step 14: Conversation Memory**
- Remember previous questions in the session
- Handle follow-up questions ("What about Q2?")
- Use AgentSession from agent_framework
- 🔨 Build: multi-turn RAG chatbot

**Step 15: Multi-Document Support**
- Upload multiple PDFs/files
- Search across ALL documents
- Filter by document/category
- 🔨 Build: document manager (add/delete/list)

**Step 16: Web UI (Streamlit)**
- Chat interface with message bubbles
- File upload sidebar
- Source citations with expandable sections
- 🔨 Build: full Streamlit chat app

**Step 17: Evaluation & Testing**
- Create test question-answer pairs
- Measure retrieval accuracy
- Measure answer quality
- Compare: naive vs hybrid vs CRAG vs agentic
- 🔨 Build: evaluation dashboard

**Tools:** Streamlit, agent_framework, ChromaDB
**GPU Needed:** No

---

## PART 2: IMAGE GENERATION & EDITING (Month 3-4)

### Week 9-10: How Neural Networks See Images

**Goal:** Understand how AI processes pixels

**Step 18: Image Basics in Python**
- Load images with Pillow (PIL)
- Pixels: each pixel = (R, G, B) = 3 numbers
- Image = 3D array: height × width × 3
- 🔨 Build: basic image editor with Pillow

**Step 19: Convolutional Neural Networks (CNN)**
- Convolution: filter slides over image, detects edges/shapes
- Pooling: shrink image, keep important parts
- Stack layers: edges → shapes → objects → scenes
- 🔨 Build: image classifier with PyTorch CNN (CIFAR-10)

**Step 20: Transfer Learning**
- Use pre-trained ResNet/EfficientNet
- Fine-tune on YOUR images (only need 100 images)
- Feature extraction: CNN as "image embedder"
- 🔨 Build: custom image classifier

**Tools:** PyTorch, torchvision, Pillow
**GPU Needed:** Yes (GTX 1650 works ✅)

---

### Week 11-12: Diffusion Models From Scratch

**Goal:** Build a mini DALL-E to understand how it works

**Step 21: Diffusion on 2D Points**
- Smiley face points → add noise → denoise → new smiley
- Simplest possible diffusion (50 lines of code)
- 🔨 Build: 2D point diffusion

**Step 22: Diffusion on Tiny Images (28×28)**
- MNIST digits (0-9)
- Build noise scheduler (linear, cosine)
- Build U-Net denoiser
- 🔨 Build: generate handwritten digits from noise

**Step 23: Add Text Conditioning**
- "Generate a 7" → model generates image of 7
- Class conditioning → text conditioning (CLIP)
- Classifier-free guidance
- 🔨 Build: conditional digit generator

**Step 24: Understanding the Math**
- Forward process: q(x_t | x_0)
- Reverse process: p(x_{t-1} | x_t)
- Loss function: predict noise
- 🔨 Build: implement each formula, visualize each step

**Tools:** PyTorch, matplotlib
**GPU Needed:** Yes (GTX 1650 fine for small images)
**Resource:** Hugging Face Diffusion Course (free)

---

### Week 13-14: Stable Diffusion — Generate & Edit

**Goal:** Use production-quality image generation

**Step 25: Run Stable Diffusion Locally**
- Install diffusers (Hugging Face)
- SD 1.5 fits in 4GB with: float16 + attention_slicing
- 🔨 Build: local text-to-image generator

**Step 26: Image-to-Image (img2img)**
- Photo + prompt + strength → modified photo
- Style transfer: "make this look like a painting"
- 🔨 Build: photo style transformer

**Step 27: Inpainting**
- Photo + mask + prompt → edit only masked region
- "Change only the sky to sunset"
- 🔨 Build: selective image editor

**Step 28: Outpainting**
- Extend images beyond their borders
- AI imagines what's beyond the frame
- 🔨 Build: image extender tool

**Tools:** diffusers, transformers, Pillow
**GPU Needed:** Yes (GTX 1650 with optimizations ✅)

---

### Week 15-16: Advanced Image Editing

**Goal:** Precise control over image generation

**Step 29: ControlNet**
- Guide generation with structure (pose, edges, depth)
- Stick figure → realistic person in same pose
- 🔨 Build: pose-guided image generator

**Step 30: LoRA (Low-Rank Adaptation)**
- Fine-tune SD on 10-20 images of YOUR subject
- Tiny adapter (4MB vs 4GB full model)
- 🔨 Build: train LoRA of your face/art style

**Step 31: IP-Adapter (Image Prompt)**
- Use reference image as part of prompt
- Combine reference style with text
- 🔨 Build: style-reference image generator

**Step 32: AI Image Editor Agent**
- LLM + all image tools combined
- "Make the sky dramatic" → agent picks right tool
- 🔨 Build: full AI image editing agent

**Tools:** diffusers, controlnet, agent_framework
**GPU Needed:** Yes (4GB tight but works with LoRA)

---

## PART 3: VIDEO AI (Month 5-6)

### Week 17-18: Video Fundamentals

**Goal:** Understand video = sequence of images + time

**Step 33: Video Basics in Python**
- Video = 30 frames per second
- Extract/process/re-assemble frames
- 🔨 Build: basic video processor

**Step 34: Object Detection in Video**
- YOLO: detect objects in each frame
- Track objects across frames
- 🔨 Build: video object tracker

**Step 35: Video Understanding with Vision AI**
- Sample key frames → send to GPT-4o
- Auto-generate timestamps & descriptions
- 🔨 Build: auto video summarizer

**Tools:** OpenCV, ultralytics (YOLO), moviepy
**GPU Needed:** Yes for YOLO

---

### Week 19-20: Video Generation

**Goal:** Understand Sora-like video generation

**Step 36: Frame Interpolation**
- Generate in-between frames (slow-mo effect)
- 24fps → 60fps using optical flow
- 🔨 Build: video frame interpolator

**Step 37: Animated Image Generation**
- AnimateDiff: make SD images move
- Generate 2-4 second clips from text
- 🔨 Build: text-to-short-video generator

**Step 38: Video Diffusion Concepts**
- Spacetime patches (height × width × time)
- Video VAE, temporal consistency
- Study Sora architecture (needs massive GPU)
- 🔨 Build: understand & diagram the architecture

**Tools:** diffusers, AnimateDiff, OpenCV
**GPU Needed:** Yes (AnimateDiff needs 6GB+ → use Colab)

---

### Week 21-22: Video Editing with AI

**Goal:** Build AI-powered video editing tools

**Step 39: AI-Powered Video Editing**
- Auto-cut silence & dead air
- Smart trim: keep only relevant parts
- Auto-captions with Whisper
- 🔨 Build: auto video editor

**Step 40: Background Replacement**
- Segment person from background per frame
- Replace with image/video
- 🔨 Build: video background replacer

**Step 41: Style Transfer on Video**
- Apply artistic style to every frame
- Temporal coherence (no flickering)
- 🔨 Build: video style transformer

**Step 42: AI Video Effects**
- Object removal from video
- Face swap (consistent across frames)
- 🔨 Build: AI video effects toolkit

**Tools:** OpenCV, moviepy, Whisper, rembg, diffusers
**GPU Needed:** Yes (Colab for heavy tasks)

---

### Week 23-24: Capstone Project

**Build: AI Video Production Assistant**
- Upload raw video → AI edits automatically
- Auto-cut, captions, scene detection
- RAG: search video library by content
- Chat: "Make the intro shorter"
- Multi-agent workflow: Researcher → Editor → Enhancer → Exporter

---

## TOOLS & LIBRARIES SUMMARY

### Phase 1 — RAG:
```
pip install sentence-transformers chromadb rank-bm25
pip install pymupdf streamlit
```

### Phase 2 — Images:
```
pip install torch torchvision (with CUDA)
pip install diffusers transformers accelerate
pip install controlnet-aux
```

### Phase 3 — Video:
```
pip install opencv-python moviepy
pip install openai-whisper ultralytics
pip install rembg
```

---

## FREE LEARNING RESOURCES

1. **Andrej Karpathy — Neural Networks: Zero to Hero** (YouTube)
   https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

2. **Fast.ai — Practical Deep Learning** (free course)
   https://course.fast.ai

3. **Hugging Face — Diffusion Models Course** (free)
   https://huggingface.co/learn/diffusion-course

4. **Google Colab** — Free GPU in browser
   https://colab.research.google.com

5. **Kaggle Notebooks** — Free GPU, 30 hrs/week
   https://kaggle.com

---

## YOUR SETUP

- **Python:** 3.13.12
- **GPU:** NVIDIA GTX 1650 (4GB VRAM)
- **Package Manager:** uv v0.10.2
- **AI Model:** Azure OpenAI gpt-4.1-mini
- **Agent Framework:** agent-framework v1.0.0b260212
- **Project Location:** c:\Users\mohit\OneDrive\Desktop\ai-agent
- **GitHub:** https://github.com/mohitsamant2k/ai-agent

---

## WHAT COMPANIES HIRE FOR (2026)

1. RAG pipelines — Part 1 covers this
2. Vector databases — Part 1 covers this
3. Agent orchestration — Part 1 covers this
4. Tool calling — Already done ✅
5. LLM prompt engineering — All projects teach this
6. Multi-agent systems — Part 1 & 2 cover this
7. Production deployment — Phase 5 of any project

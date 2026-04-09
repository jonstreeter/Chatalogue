# Future Features

## Reference-Grounded Personality Training

**Goal:** Allow users to supply reference texts (books, articles, debate transcripts) that a speaker draws on for their arguments, so the trained LoRA can reproduce their logical reasoning and characteristic references — without contaminating speaking style.

### Approach: Synthetic Examples + RAG

**Training-time (synthetic example generation):**
- User uploads reference texts to an avatar's reference library
- At dataset build time, chunk reference texts into passage-sized pieces
- Use Ollama to generate synthetic training examples: given a passage and the speaker's style profile, produce a response as the speaker would naturally discuss that topic
- Tag these as `source_kind: "reference_grounded"` so they can be scored, reviewed, and ratio-controlled alongside real transcript examples
- Cap synthetic-to-real ratio (suggested: no more than 20-30% of total training set) to prevent the model from drifting toward the generation LLM's voice

**Inference-time (RAG retrieval):**
- Embed reference texts into a vector store (per-avatar)
- During test chat, retrieve relevant passages and inject them as grounding context in the system prompt
- The personality LoRA handles style; retrieved passages handle knowledge the model may not have internalized

### Why both approaches

- Synthetic examples bake in durable knowledge — the model can reference concepts without retrieval
- RAG covers the long tail: new material, niche references, and topics where synthetic coverage is sparse
- Together they provide reliable grounding without requiring the user to retrain every time they add a new source

### Key design considerations

- Synthetic examples need aggressive quality filtering — the generation LLM may not nail the speaker's voice, so the existing scoring pipeline (style_score, substance_score) should evaluate them like any other example
- Reference texts should be chunked with overlap to preserve argument context across chunk boundaries
- The ratio of reference-grounded examples should be user-configurable in the training config
- Need a UI for managing the reference library: upload, preview chunks, include/exclude individual passages

### Alternative considered: Two-stage LoRA with layer separation

Train a knowledge LoRA targeting only MLP layers (gate/up/down_proj) on reference texts, and a personality LoRA targeting only attention layers (q/v/o_proj) on transcripts. Merge at inference via PEFT adapter composition. Theoretically clean but in practice the attention/MLP knowledge-style boundary is blurry, and managing two adapters adds significant complexity. The synthetic+RAG approach achieves the same goal with less infrastructure.

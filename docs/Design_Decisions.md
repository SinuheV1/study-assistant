# Design Decisions & Tradeoffs

## Why Not NotebookLM?
- No architecture control
- No embedding tuning
- No system-level learning
- No resume signal

## Why Local-First?
- Cost control
- Privacy
- Architectural clarity
- Engineering discipline

## Why Optional API Escalation?
- Higher quality when needed
- Controlled cost
- Comparative evaluation

## Why Pure Python?
- Deep understanding of RAG internals
- Avoid over-reliance on frameworks
- MLE-relevant experience

## Why Avoid Early Automation?
- Reduces maintenance burden
- Keeps system focused
- Avoids hype traps
- Increases reliability

## Why Mac mini Over Raspberry Pi?
- Enough memory for embeddings + LLM
- Better dev experience
- Pi reserved for infra learning later
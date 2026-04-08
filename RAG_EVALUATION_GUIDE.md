# RAG Pipeline Evaluation Guide

How to measure and improve the quality of your retrieval and generation pipeline, from individual components to end-to-end evaluation.

---

## Table of Contents

1. [What to Evaluate](#what-to-evaluate)
2. [Evaluation Framework Overview](#evaluation-framework-overview)
3. [Method 1: LLM-as-Judge](#method-1-llm-as-judge)
4. [Method 2: RAGAS (Automated RAG Assessment)](#method-2-ragas)
5. [Method 3: Component-Level Metrics](#method-3-component-level-metrics)
6. [Method 4: Human Evaluation](#method-4-human-evaluation)
7. [Method 5: A/B Testing in Production](#method-5-ab-testing-in-production)
8. [Method 6: DeepEval Framework](#method-6-deepeval-framework)
9. [Method 7: Synthetic Test Set Generation](#method-7-synthetic-test-set-generation)
10. [Building Your Evaluation Dataset](#building-your-evaluation-dataset)
11. [Putting It All Together](#putting-it-all-together)

---

## What to Evaluate

Your pipeline has 6 stages. Each needs different metrics:

```
User Question
    │
    ▼
┌──────────────────┐
│ 1. EMBEDDING     │ ← Is the query vector accurate?
│    (nemotron)    │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│SEMANTIC│ │BM25    │ ← Are the RIGHT chunks retrieved?
│SEARCH  │ │KEYWORD │    (Retrieval quality)
└───┬────┘ └───┬────┘
    └────┬─────┘
         ▼
┌──────────────────┐
│ 4. RRF FUSION    │ ← Does fusion improve over individual methods?
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 5. RERANKING     │ ← Are the top-5 the most relevant?
│    (MiniLM)      │    (Reranking quality)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 6. ANSWER        │ ← Is the answer correct, grounded, complete?
│    SYNTHESIS     │    (Generation quality)
│    (LLM API)     │
└──────────────────┘
```

### The Three Pillars of RAG Evaluation

```
                    ┌──────────────────────────┐
                    │   RAG EVALUATION          │
                    │                          │
                    │  1. RETRIEVAL QUALITY     │  Did we find the right chunks?
                    │  2. GENERATION QUALITY    │  Is the answer correct?
                    │  3. END-TO-END QUALITY    │  Does the user get value?
                    └──────────────────────────┘
```

---

## Evaluation Framework Overview

| Method | What It Measures | Cost | Effort | Best For |
|--------|-----------------|------|--------|----------|
| **LLM-as-Judge** | Answer quality, faithfulness | API cost (~$5-20/run) | Low | Fast iteration, no labeled data needed |
| **RAGAS** | Retrieval + generation (4 metrics) | API cost | Low | Comprehensive automated eval |
| **Component Metrics** | Precision, recall, MRR per stage | Free | Medium | Debugging which stage is weak |
| **Human Evaluation** | Ground truth quality | Free but slow | High | Gold standard, final validation |
| **A/B Testing** | User satisfaction in production | Free | Medium | Production optimization |
| **DeepEval** | 14+ metrics with LLM scoring | API cost | Low | Most comprehensive automated tool |
| **Synthetic Test Gen** | Auto-creates eval dataset | API cost | Low | When you have no test data |

---

## Method 1: LLM-as-Judge

Use a strong LLM (GPT-4o, Claude) to evaluate your pipeline's answers. The judge LLM should be **stronger** than your answer synthesis LLM.

### How It Works

```
Your pipeline:
  Question: "What is the SLA for provisioning?"
  Retrieved chunks: [chunk_42, chunk_15, chunk_88]
  Generated answer: "The SLA for provisioning is 4 hours for standard
                     requests and 1 hour for priority requests."

Judge LLM evaluates 4 dimensions:
  Faithfulness:  Is the answer supported by the retrieved chunks?  → 9/10
  Relevance:     Does the answer actually address the question?     → 10/10
  Completeness:  Does it cover all aspects of the question?         → 7/10
  Correctness:   Is the information factually correct?              → 9/10
```

### Judge Prompts

#### Faithfulness (Is the answer grounded in the context?)

```
You are evaluating a RAG system's answer for faithfulness.

CONTEXT (retrieved chunks):
{chunks}

QUESTION: {question}

ANSWER: {generated_answer}

Rate the FAITHFULNESS of the answer on a scale of 1-5:
1 = Completely hallucinated — contains claims not in the context
2 = Mostly hallucinated — some facts from context, but key claims are invented
3 = Partially faithful — mix of grounded and ungrounded claims
4 = Mostly faithful — nearly all claims are in the context, minor additions
5 = Fully faithful — every claim can be traced to the context

Provide:
- Score (1-5)
- List each claim in the answer and whether it appears in the context
- Any hallucinated claims
```

#### Answer Relevance (Does it answer the question?)

```
You are evaluating whether a RAG system's answer is relevant to the question.

QUESTION: {question}

ANSWER: {generated_answer}

Rate the RELEVANCE on a scale of 1-5:
1 = Completely irrelevant — doesn't address the question at all
2 = Tangentially related — discusses the topic but doesn't answer
3 = Partially relevant — answers part of the question
4 = Mostly relevant — answers the question with minor tangents
5 = Perfectly relevant — directly and completely answers the question

Provide:
- Score (1-5)
- What parts of the question were answered
- What parts were missed or addressed incorrectly
```

#### Context Relevance (Did retrieval find the right chunks?)

```
You are evaluating whether the retrieved context is relevant to the question.

QUESTION: {question}

RETRIEVED CONTEXT:
[Chunk 1]: {chunk_1}
[Chunk 2]: {chunk_2}
...

For each chunk, rate its relevance to answering the question:
- RELEVANT: Contains information needed to answer the question
- PARTIALLY RELEVANT: Contains some useful information
- IRRELEVANT: Not useful for answering the question

Provide:
- Per-chunk relevance label
- Overall retrieval precision (% of chunks that are relevant)
- Any information needed to answer that is MISSING from the chunks
```

### Running LLM-as-Judge

```python
# Pseudocode for evaluation harness:

eval_dataset = [
    {"question": "What is the SLA?", "ground_truth": "4 hours standard, 1 hour priority"},
    {"question": "What is ERR-4102?", "ground_truth": "Timeout error in provisioning"},
    # ... 50-100 questions
]

results = []
for item in eval_dataset:
    # Run your pipeline
    chunks = hybrid_search(item["question"])
    reranked = reranker.rerank(item["question"], chunks)
    answer = synthesizer.synthesize(item["question"], reranked)
    
    # Judge with GPT-4o
    faithfulness = judge_faithfulness(chunks, answer)
    relevance = judge_relevance(item["question"], answer)
    correctness = judge_correctness(answer, item["ground_truth"])
    
    results.append({
        "question": item["question"],
        "faithfulness": faithfulness.score,
        "relevance": relevance.score,
        "correctness": correctness.score,
    })

# Aggregate scores
avg_faithfulness = mean([r["faithfulness"] for r in results])  # Target: > 4.0
avg_relevance = mean([r["relevance"] for r in results])        # Target: > 4.0
avg_correctness = mean([r["correctness"] for r in results])    # Target: > 3.5
```

---

## Method 2: RAGAS (Automated RAG Assessment)

RAGAS is a framework specifically designed for RAG evaluation. It computes 4 metrics automatically using an LLM.

### The 4 RAGAS Metrics

```
                        ┌────────────────────────┐
                        │    RAGAS METRICS        │
                        └────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
     │ RETRIEVAL METRICS│  │ GENERATION      │  │ END-TO-END     │
     │                  │  │ METRICS          │  │                │
     │ Context          │  │ Faithfulness     │  │ Answer         │
     │ Precision        │  │ (are answers     │  │ Correctness    │
     │ (are retrieved   │  │  grounded in     │  │ (is the final  │
     │  chunks useful?) │  │  retrieved       │  │  answer right?)│
     │                  │  │  context?)       │  │                │
     │ Context          │  │                  │  │                │
     │ Recall           │  │ Answer           │  │                │
     │ (did we find all │  │ Relevance        │  │                │
     │  needed chunks?) │  │ (does it answer  │  │                │
     │                  │  │  the question?)  │  │                │
     └─────────────────┘  └─────────────────┘  └────────────────┘
```

### How Each Metric Works

#### 1. Faithfulness (0-1)

```
"Can every claim in the answer be traced back to the retrieved context?"

Process:
  1. LLM extracts all claims from the answer:
     Answer: "The SLA is 4 hours. Priority tickets get 1 hour response."
     Claims: ["SLA is 4 hours", "Priority tickets get 1 hour response"]
  
  2. For each claim, LLM checks if it's supported by context:
     "SLA is 4 hours" → Found in chunk_42 → SUPPORTED ✅
     "Priority tickets get 1 hour response" → Found in chunk_15 → SUPPORTED ✅
  
  3. Score = supported_claims / total_claims = 2/2 = 1.0

Low faithfulness (< 0.7) = Your LLM is hallucinating
Fix: Improve prompt, use stronger model, or add "cite your sources"
```

#### 2. Answer Relevance (0-1)

```
"Does the answer address what was actually asked?"

Process:
  1. LLM generates N questions that the ANSWER would be a good response to:
     Answer: "The SLA is 4 hours for standard provisioning."
     Generated questions:
       - "What is the SLA for provisioning?"
       - "How long does standard provisioning take?"
       - "What is the response time for service requests?"
  
  2. Compare generated questions with the ORIGINAL question via embedding similarity:
     Original: "What is the SLA for provisioning?"
     Similarity scores: [0.95, 0.82, 0.71]
  
  3. Score = mean(similarities) = 0.83

Low relevance (< 0.6) = Answer is tangential or off-topic
Fix: Better retrieval (the right chunks weren't found) or better prompt
```

#### 3. Context Precision (0-1)

```
"Are the top-ranked chunks actually useful for answering?"

Process:
  Ranked chunks:     [chunk_42, chunk_15, chunk_200, chunk_88, chunk_301]
  Useful for answer?  [  YES  ,   YES  ,    NO    ,   YES  ,    NO   ]
  
  Precision@1 = 1/1 = 1.0  (chunk_42 is useful)
  Precision@2 = 2/2 = 1.0  (chunk_42 + chunk_15 are useful)
  Precision@3 = 2/3 = 0.67 (chunk_200 is NOT useful)
  Precision@4 = 3/4 = 0.75
  Precision@5 = 3/5 = 0.60
  
  Average Precision = mean of precision@k where chunk is useful
                    = (1.0 + 1.0 + 0.75) / 3 = 0.92

Low precision (< 0.5) = Retrieval is finding irrelevant chunks
Fix: Better embeddings, tune HNSW, improve BM25 tokenization
```

#### 4. Context Recall (0-1)

```
"Did we retrieve ALL the chunks needed to answer the question?"

Process:
  Ground truth answer: "SLA is 4 hours standard, 1 hour priority.
                        Escalation happens after 2 failed attempts."
  
  Ground truth claims:
    1. "SLA is 4 hours standard"              → Found in chunk_42 ✅
    2. "1 hour priority"                       → Found in chunk_15 ✅
    3. "Escalation after 2 failed attempts"    → NOT in any chunk ❌
  
  Recall = 2/3 = 0.67

Low recall (< 0.7) = Missing important chunks
Fix: Increase k (retrieve more), improve embedding model, tune BM25
```

### Using RAGAS

```python
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare your evaluation data
eval_data = {
    "question": [
        "What is the SLA for provisioning?",
        "What is error ERR-4102?",
        # ... more questions
    ],
    "answer": [
        "The SLA is 4 hours...",          # Generated by YOUR pipeline
        "ERR-4102 is a timeout error...", # Generated by YOUR pipeline
    ],
    "contexts": [
        ["chunk_42 text...", "chunk_15 text..."],   # Retrieved by YOUR pipeline
        ["chunk_103 text...", "chunk_42 text..."],
    ],
    "ground_truth": [
        "The SLA for standard provisioning is 4 hours, priority is 1 hour.",
        "ERR-4102 is a timeout error in the provisioning subsystem.",
    ],
}

dataset = Dataset.from_dict(eval_data)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(results)
# {'faithfulness': 0.87, 'answer_relevancy': 0.91, 
#  'context_precision': 0.82, 'context_recall': 0.75}
```

---

## Method 3: Component-Level Metrics

Evaluate each stage independently to find bottlenecks.

### Retrieval Metrics (No LLM Needed)

```
For each test question, you need:
  - Query
  - Ground truth relevant chunk IDs (manually labeled)
  - Retrieved chunk IDs (from your pipeline)

Metrics:

  Recall@K:
    "Of all relevant chunks, what % did we find in top K?"
    relevant_in_top_k / total_relevant
    
    Example: 3 relevant chunks exist. Top-10 found 2 of them.
    Recall@10 = 2/3 = 0.67

  Precision@K:
    "Of the top K retrieved, what % are actually relevant?"
    relevant_in_top_k / k
    
    Example: Top-5 retrieved, 3 are relevant.
    Precision@5 = 3/5 = 0.60

  MRR (Mean Reciprocal Rank):
    "What's the average position of the first relevant result?"
    
    Query 1: First relevant at rank 1 → 1/1 = 1.0
    Query 2: First relevant at rank 3 → 1/3 = 0.33
    Query 3: First relevant at rank 2 → 1/2 = 0.5
    MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61

  NDCG@K (Normalized Discounted Cumulative Gain):
    "Are the MOST relevant chunks ranked HIGHEST?"
    Accounts for graded relevance (highly relevant > somewhat relevant)
```

### Comparing Search Methods

```
Run the SAME test set through each search method:

                    Recall@10    Precision@5    MRR
  BM25 only           0.62         0.55       0.58
  Semantic only        0.74         0.68       0.71
  Hybrid (RRF)         0.85         0.79       0.83
  Hybrid + Rerank      0.85         0.91       0.89  ← your pipeline

This proves each component adds value:
  BM25 → Semantic:    +19% recall (semantic understands meaning)
  Semantic → Hybrid:  +15% recall (BM25 catches exact terms)
  Hybrid → Reranked:  +15% precision (reranker filters noise)
```

### Chunking Quality Metrics

```
Test whether your 800/100 chunking preserves information:

  Chunk Coherence:
    "Does each chunk make sense as a standalone unit?"
    LLM scores each chunk 1-5 on self-containedness.
    Target: > 3.5

  Boundary Quality:
    "Do chunk boundaries split important information?"
    Check: Are image references ever split across chunks?
    Check: Are table rows split mid-row?
    Target: 0% split image tags (your SmartChunker handles this)
```

---

## Method 4: Human Evaluation

The gold standard — but expensive and slow.

### Setup

```
1. Create 50-100 test questions from your actual documents
2. Run your pipeline to generate answers
3. Have 2-3 evaluators rate each answer on:

   ┌────────────────────────────────────────────────────┐
   │ Dimension      │ Scale │ Question to Evaluator     │
   ├────────────────┤───────┤───────────────────────────┤
   │ Correctness    │ 1-5   │ Is the answer factually   │
   │                │       │ correct?                  │
   │ Completeness   │ 1-5   │ Does it cover everything  │
   │                │       │ the user would need?      │
   │ Conciseness    │ 1-5   │ Is it appropriately       │
   │                │       │ brief, without fluff?     │
   │ Groundedness   │ Yes/No│ Does it only use info     │
   │                │       │ from the documents?       │
   │ Helpfulness    │ 1-5   │ Would this answer satisfy │
   │                │       │ the user?                 │
   └────────────────┴───────┴───────────────────────────┘

4. Calculate Inter-Annotator Agreement (Cohen's Kappa)
   Target: κ > 0.6 (substantial agreement)
```

### Side-by-Side Comparison

```
Show evaluators answers from different pipeline configs:

  Config A: BM25 only + no reranking
  Config B: Semantic only + reranking
  Config C: Hybrid + reranking (your current pipeline)

  Question: "What is the escalation process?"

  Answer A: "The escalation process involves contacting support..."
  Answer B: "For escalation, the workflow follows three stages: 1) Initial
             triage within 1 hour, 2) Level 2 review within 4 hours..."
  Answer C: "The escalation process has 3 stages: initial triage (1 hr SLA),
             Level 2 review (4 hr SLA), and management escalation (24 hr SLA).
             See the flowchart on page 15 for the visual workflow."

  Evaluator picks: C wins (includes SLA numbers + references the image)
```

---

## Method 5: A/B Testing in Production

Once deployed, test changes with real users.

```
Setup:
  50% of traffic → Pipeline A (current)
  50% of traffic → Pipeline B (experimental change)

Metrics to track:
  ┌──────────────────────────┬──────────────────────────┐
  │ Implicit Signals         │ Explicit Signals          │
  ├──────────────────────────┼──────────────────────────┤
  │ Time on answer page      │ Thumbs up/down button    │
  │ Did user ask follow-up?  │ "Was this helpful?" Y/N  │
  │ Did user rephrase query? │ User feedback text       │
  │ Click-through on sources │ Support ticket filed?    │
  └──────────────────────────┴──────────────────────────┘

  User rephrased query → Original answer was probably bad
  User clicked sources → Answer may have been incomplete, user wanted more
  Thumbs down → Directly bad answer
  
  Statistical significance: Run for ~500 queries per variant before deciding.
```

---

## Method 6: DeepEval Framework

The most comprehensive automated evaluation library. Supports 14+ metrics.

```python
# pip install deepeval

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
)

# Create test cases
test_case = LLMTestCase(
    input="What is the SLA for provisioning?",
    actual_output="The SLA is 4 hours for standard requests...",
    expected_output="Standard provisioning SLA is 4 hours, priority is 1 hour.",
    retrieval_context=["chunk_42 text...", "chunk_15 text..."],
)

# Run all metrics
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    HallucinationMetric(threshold=0.5),
]

evaluate(test_cases=[test_case], metrics=metrics)
```

### DeepEval vs RAGAS Comparison

| Feature | RAGAS | DeepEval |
|---------|-------|---------|
| **Core metrics** | 4 | 14+ |
| **Hallucination detection** | Via faithfulness | Dedicated metric |
| **Toxicity/Bias** | ❌ | ✅ |
| **Latency tracking** | ❌ | ✅ |
| **Cost tracking** | ❌ | ✅ |
| **CI/CD integration** | Manual | `deepeval test run` built-in |
| **Dashboard** | ❌ | ✅ (Confident AI platform) |
| **Custom metrics** | ✅ | ✅ |
| **Ease of use** | Simpler | More features, steeper curve |

---

## Method 7: Synthetic Test Set Generation

**Problem:** You need 50-100 test questions, but creating them manually is tedious.

**Solution:** Use an LLM to generate question-answer pairs FROM your documents.

```
Process:
  1. Take your ingested document chunks
  2. Feed them to GPT-4o with this prompt:
  
  ┌───────────────────────────────────────────────────────────┐
  │  Given this document excerpt, generate 3 types of         │
  │  question-answer pairs:                                   │
  │                                                           │
  │  1. FACTUAL: A question with a specific, factual answer   │
  │     "What is the SLA for standard provisioning?"          │
  │     → "4 hours"                                           │
  │                                                           │
  │  2. REASONING: A question requiring synthesis across      │
  │     multiple facts in the context                         │
  │     "How does priority affect provisioning timelines?"    │
  │     → "Priority requests have a 1-hour SLA vs 4 hours"   │
  │                                                           │
  │  3. MULTI-HOP: A question needing info from multiple      │
  │     chunks (harder for retrieval)                         │
  │     "What happens if provisioning exceeds both the SLA    │
  │      and the escalation window?"                          │
  │     → "After SLA breach, escalation triggers..."          │
  └───────────────────────────────────────────────────────────┘

  3. Human reviews and filters (remove bad/ambiguous questions)
  4. Result: 50-100 high-quality test pairs with ground truth
```

### Tools for Synthetic Generation

```
RAGAS Test Set Generator:
  from ragas.testset.generator import TestsetGenerator
  generator = TestsetGenerator.from_langchain(llm, embeddings)
  testset = generator.generate_with_langchain_docs(documents, test_size=50)
  # Automatically creates simple, reasoning, and multi-context questions

Giskard (open source):
  Generates adversarial test cases designed to break your pipeline.
  Tests for edge cases: negation, ambiguity, out-of-scope questions.
```

---

## Building Your Evaluation Dataset

### Minimum Viable Evaluation Set

```
For your pipeline, create at least:

  ┌─────────────────────────────────────────────────────────────┐
  │ Category               │ Count │ Example                    │
  ├────────────────────────┼───────┼────────────────────────────┤
  │ Factual (exact answer) │  20   │ "What is the SLA?"         │
  │ Reasoning (synthesis)  │  15   │ "Compare standard vs       │
  │                        │       │  priority provisioning"    │
  │ Multi-chunk            │  10   │ "Summarize all error       │
  │                        │       │  codes and their causes"   │
  │ Image-dependent        │   5   │ "What does the flowchart   │
  │                        │       │  on page 15 show?"         │
  │ Keyword-specific       │  10   │ "What is ERR-4102?"        │
  │ Out-of-scope           │   5   │ "What is the weather?"     │
  │ Ambiguous              │   5   │ "Tell me about policies"   │
  │ Adversarial            │   5   │ "The SLA is 24 hours,      │
  │                        │       │  right?" (it's 4 hours)    │
  └────────────────────────┴───────┴────────────────────────────┘
  
  Total: ~75 test cases (minimum viable)
  Target: 100-200 for robust evaluation
```

### Test Case Format

```json
{
  "id": "test_001",
  "question": "What is the SLA for standard provisioning?",
  "ground_truth_answer": "The SLA for standard provisioning is 4 hours.",
  "ground_truth_chunks": ["chunk_042", "chunk_043"],
  "category": "factual",
  "difficulty": "easy",
  "requires_image": false
}
```

---

## Putting It All Together

### Recommended Evaluation Strategy

```
Phase 1: Quick Baseline (Day 1)
  ├── Generate 50 synthetic test questions (RAGAS TestsetGenerator)
  ├── Run LLM-as-Judge on all 50 (faithfulness + relevance)
  ├── Record baseline scores
  └── Time: ~2 hours

Phase 2: Component Analysis (Day 2-3)
  ├── Manually label 30 questions with ground-truth relevant chunks
  ├── Measure Recall@10, Precision@5, MRR for:
  │   ├── BM25 only
  │   ├── Semantic only
  │   ├── Hybrid (BM25 + Semantic)
  │   └── Hybrid + Reranked
  ├── Identify weakest component
  └── Time: ~4 hours

Phase 3: Comprehensive Eval (Week 1-2)
  ├── Build 100-question eval dataset (synthetic + manual)
  ├── Run RAGAS or DeepEval full suite
  ├── Human evaluation on 30 critical questions
  ├── Establish CI/CD eval pipeline (auto-run on code changes)
  └── Time: ~1 week

Phase 4: Production Monitoring (Ongoing)
  ├── A/B test pipeline changes with real users
  ├── Track implicit signals (rephrasing, follow-ups)
  ├── Weekly eval runs to detect drift
  └── Quarterly eval dataset refresh
```

### Target Scores

```
╔══════════════════════════════════════════════════════╗
║  METRIC                    │ MINIMUM  │ GOOD  │ GREAT ║
╠════════════════════════════╪══════════╪═══════╪═══════╣
║  Faithfulness              │  > 0.70  │ > 0.85│ > 0.95║
║  Answer Relevance          │  > 0.65  │ > 0.80│ > 0.90║
║  Context Precision         │  > 0.50  │ > 0.75│ > 0.85║
║  Context Recall            │  > 0.60  │ > 0.75│ > 0.90║
║  Retrieval Recall@10       │  > 0.60  │ > 0.80│ > 0.90║
║  Retrieval Precision@5     │  > 0.50  │ > 0.70│ > 0.85║
║  MRR                       │  > 0.50  │ > 0.70│ > 0.85║
║  Human Correctness (1-5)   │  > 3.0   │ > 4.0 │ > 4.5 ║
║  Hallucination Rate        │  < 30%   │ < 15% │ < 5%  ║
╚══════════════════════════════════════════════════════╝
```

### Diagnostic Flowchart

```
Low scores? Use this to find the root cause:

  Low Faithfulness (< 0.7)?
    → LLM is hallucinating
    → Fix: Stronger system prompt, add "only use provided context"
    → Fix: Use a more instruction-following LLM
    → Fix: Reduce temperature to 0

  Low Context Recall (< 0.6)?
    → Retrieval is missing important chunks
    → Fix: Increase k (retrieve top 30 instead of 20)
    → Fix: Better embedding model
    → Fix: Lower chunk size (more granular chunks)

  Low Context Precision (< 0.5)?
    → Retrieval is finding irrelevant chunks
    → Fix: Better embedding model
    → Fix: Improve BM25 tokenization (stemming, stopwords)
    → Fix: Tune RRF alpha parameter

  Low Answer Relevance (< 0.6)?
    → Answer doesn't address the question
    → Fix: Better system prompt
    → Fix: Feed more relevant chunks (improve retrieval first)

  High Hallucination (> 20%)?
    → Check faithfulness
    → Fix: Add explicit "do not add information not in context" to prompt
    → Fix: Use smaller, more constrained model
    → Fix: Add post-generation fact-checking step
```

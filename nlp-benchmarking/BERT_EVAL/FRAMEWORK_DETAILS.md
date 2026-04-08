# Setup Overview: BERT Evaluation & Profiling Pipeline
The BERT Evaluation & Profiling Pipeline is a specialized analytical framework designed to bridge the gap between high-level linguistic accuracy and low-level hardware efficiency for Transformer-based architectures. While modern LLMs often prioritize generative capabilities, this pipeline focuses on Natural Language Understanding (NLU), providing a dual-layered audit of how encoder-only and encoder-decoder models process structured information.

## Core Objectives
- Multidimensional NLU Validation: The framework quantifies model intelligence across four distinct cognitive tasks: extractive reasoning (SQuAD), linguistic probability (Wikitext-2/Perplexity), sentiment classification (SST2), and entity recognition (CoNLL-2003).
- Micro-Architectural Telemetry: Beyond simple benchmarks, the pipeline extracts granular data on the execution of the Transformer blocks. By capturing MNK dimensions ($M, N, K$ parameters) and cuBLAS function calls, it identifies the exact mathematical "footprint" of the model's self-attention and feed-forward layers.
- Operational Benchmarking: The setup differentiates between Open-Source vs. Closed-Source kernels, providing a clear picture of how much performance is derived from standardized libraries versus specialized, vendor-specific GPU optimizations.

## Evaluation Paradigms
The framework rigorously tests the "Encoder" functionality through several theoretical lenses:
- Contextual Extraction & Abstention (Question Answering): Utilizing SQuAD v1.1/v2.0, it measures the model's ability to map relationships between queries and context, specifically testing the "boundary detection" logic required to identify unanswerable questions.
- Semantic Cohesion (Masked Language Modeling): Through Perplexity analysis on Wikitext, the framework evaluates the model's internal statistical representation of language—essentially measuring how "surprised" the model is by standard human text.
- Entity & Sentiment Granularity: By benchmarking against GLUE and CoNLL-2003, the system assesses the model's ability to compress complex semantic meaning into discrete classifications, a fundamental requirement for production-grade NLU applications.

## Supported Datasets & Tasks
The pipeline is pre-configured to handle several benchmark datasets, each targeting a specific natural language understanding task:
- Question Answering (SQuAD v1.1 / v2.0): Evaluates the model's ability to extract answers from a given context. SQuAD v2.0 includes unanswerable questions to test the model's ability to refrain from answering.
- Masked Language Modeling (Wikitext-2): Measures Perplexity by masking tokens in the wikitext-2-raw-v1 validation set.
- Text Classification (GLUE/SST2): Benchmarks binary sentiment analysis using the Stanford Sentiment Treebank.
- Token Classification (CoNLL-2003): Performs Named Entity Recognition (NER) to identify persons, organizations, and locations.
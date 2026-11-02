Yes—your concern is justified. **Adding contemporary open-weight models would materially strengthen the research**, particularly if you want conclusions about local versus hosted hiring decision support.

From [the evaluation notebook](/Users/tobialao/workspace/research/hiring-dss-simulation/notebooks/inference_evaluation_new.ipynb), the current comparison is effectively:

- One Llama 3.1 8B model in FP16, Q8 and Q4.
- One hosted configuration labelled simply `openai`.

Those are **four configurations, but only two model choices**. This is useful for studying quantization, but insufficient for general claims about open-weight versus proprietary models. Also, the notebook does not establish which hosted model produced the saved results; the current runner’s default cannot establish that retrospectively.

**Models I would introduce**

These are a practical shortlist checked against current official model cards, rather than a claim that any one model is best for your task.

| Model | Research role | Why include it |
|---|---|---|
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | First priority: contemporary small-model comparator | Close in parameter count to Llama 3.1 8B, making this a more informative comparison than only adding much larger models. Apache 2.0. |
| [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it) | Independent model family | Adds another contemporary architecture and training lineage at a reasonably similar scale. Apache 2.0. |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Higher-capacity open-weight comparator | Tests whether a larger contemporary local model closes the task-performance gap. Apache 2.0. |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | Reasoning and agent-workflow comparator | Adds configurable reasoning effort and a mixture-of-experts architecture: approximately 21B total parameters, 3.6B active. Apache 2.0. |
| [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) | Optional larger-resource experiment | Adds another current family, but its 119B total parameters make it a substantially different deployment tier. Apache 2.0. |

For a manageable study, **I would retain Llama 3.1 8B and add Qwen3.5-9B, Gemma 4 12B and Qwen3.8-27B**, alongside the hosted reference. Add gpt-oss if reasoning behaviour is central to your research question.

Use “open-weight” precisely in the paper: released weights and permissive licensing do not establish that the complete training data and training process are available.

**How I would make the comparison richer**

Separate the research questions:

1. **Model choice:** Compare the selected models under a declared deployment budget, keeping documents, embeddings, retrieval configuration and task instructions fixed.
2. **Quantization:** Compare FP16/Q8/Q4 within selected models. Quantization variants should remain separate from the count of model families.
3. **Reasoning budget:** Where supported, compare reasoning settings separately, recording generated tokens, latency and cost.

For this project, I would also address these evaluation issues:

- **Establish checkpoint provenance.** The local filenames contain `hiring`; verify whether these are domain-fine-tuned models. If so, document training data and ensure evaluation candidates and questions were held out.
- **Distinguish component and system performance.** Use fixed retrieved passages to compare answer generation, then separately evaluate the complete workflow where model decisions can change retrieval and routing.
- **Strengthen scoring.** Include evidence support, unsupported claims, appropriate abstention, workflow failures and—where applicable—candidate-ranking agreement. Calibrate automated judges against a blinded human-scored subset.
- **Add hiring-specific robustness tests.** Use matched synthetic profiles that vary names or demographic cues while holding qualifications constant; test irrelevant information and instructions embedded in resumes.
- **Report uncertainty.** Use paired comparisons on identical questions, repeated runs and confidence intervals. Account for questions sharing the same candidate or test case.
- **Interpret latency as deployment performance.** A local machine versus a hosted API comparison includes hardware, serving and network effects.

There are also two concrete implementation concerns. The [local runner](/Users/tobialao/workspace/research/hiring-dss-simulation/run_llm_cpp_simulation.py:864) hardcodes `chat_format="llama-3"`, so swapping model files alone would not give a valid comparison. Each family needs its supported template; [gpt-oss explicitly requires Harmony formatting](https://huggingface.co/openai/gpt-oss-20b). In the notebook, `model_full_data.copy()` is shallow, so subsequent nested deletions also remove fields from `full_evaluation_data`, losing evidence you may want for auditing.

The stronger research framing would be: **“How do model family, capacity, quantization and inference budget affect the accuracy, evidence grounding, fairness and deployment cost of a hiring DSS?”** Your existing Llama results remain valuable as the historical baseline within that design.

"""FailureModeBench: a systematic failure-mode benchmark for VLMs.

Layers:
  config      -- registries (16 datasets, models, task families)
  taxonomy    -- the 8 failure modes F1-F8 + judge rubric
  recognition -- image-classification runner (HF datasets)
  vqa         -- VLMEvalKit driver + prediction importer
  judge       -- error extraction + LLM-judge failure-mode classification
  aggregate   -- failure-rate tables + confusion matrices
  figures     -- paper figures
"""
__version__ = "0.1.0"

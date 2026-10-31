# Model tests

**Build prompt:** Test each candidate model can fit/predict through the complete pipeline, random seeds produce repeatable results within expected tolerance, and model comparison uses identical splits. Test artifact save/load round-trip and metadata/schema mismatch handling. Avoid asserting a model wins based on a fixed metric from a small sample.

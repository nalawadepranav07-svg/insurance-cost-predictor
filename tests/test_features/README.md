# Feature pipeline tests

**Build prompt:** Verify numeric and categorical columns are routed correctly; one-hot encoding handles unseen categories; fitted transforms are learned only from training folds; output dimensions stay consistent; and inference accepts raw features in any column order only when the interface explicitly aligns by name. Check that target `charges` is excluded from feature columns.

# Processed data

**Build prompt:** Store only clearly versioned, validated datasets or split manifests created by the pipeline. Ensure all learned preprocessing (imputation, scaling, encoding) is fit inside model pipelines and CV folds, not pre-applied to the entire dataset. Include schema/version and source checksum metadata. Never commit processed data if licence or privacy does not permit.

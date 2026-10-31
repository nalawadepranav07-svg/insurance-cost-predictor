# `configs/`
`default.json` holds the dataset URL/checksum, schema and input bounds, seed, test size, 5-fold CV, tuning budget and output
paths. It is validated on load (e.g. the target may not be listed as a feature) and a copy of the schema/protocol is stored
inside every model's metadata. No secrets belong here.

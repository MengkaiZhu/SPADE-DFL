# Source and publication integrity

The experiment manifests preserve historical source paths, sizes and SHA-256 hashes. Some released Python files have edited comments, diagnostic messages or source-path/integrity adapters. Their current bytes therefore differ from the historical sources.

[publication_adaptations.json](../configs/publication_adaptations.json) records both identities for each adapted vendor file: `historical_sha256`/`historical_bytes` and `published_sha256`/`published_bytes`, with the change type. Unlisted source files retain their historical bytes. The verification wrapper checks the original identity against this record and verifies the exact published bytes; the adaptation record itself has an independent digest in `publication.py`.

Historical module-map fingerprints remain unchanged. `historical_module_hashes` first verifies current module bytes and then supplies historical identities to those compatibility checks. This is distinct from measuring current hashes. `MANIFEST.json` and `SHA256SUMS.txt` describe the actual files in this archive.

Existing result records, figure metadata and historical source indexes retain the provenance of the runs or products they describe. They are not relabelled as outputs of the edited release. Optional comparison with external historical archives is described in [Optional sources](OPTIONAL_SOURCES.md).

# Optional author sources

The release does not include the original author manuscript, multi-gigabyte archives, or private working directories. The packaged reference aggregates and their verification records remain usable without them. Tests that compare against those external sources report `skipped` when no source index is configured; they do not report a successful comparison.

Historical-source import is a direct request and therefore fails if the index or a requested source is unavailable. Configure the index in either of these ways:

```powershell
$env:SPADE_DFL_SOURCE_INDEX = '..\author-sources\source-index.json'
python tools/import_reference_results.py --output-root imported_reference_results
python tools/import_reference_results.py --source-index '..\author-sources\source-index.json' --output-root imported_reference_results
```

`--source-index` takes precedence over `SPADE_DFL_SOURCE_INDEX`. A relative `root` inside the JSON file is resolved from the directory containing that JSON file, not from the release directory or current working directory.

The documented index format is `schema_version` 1. The importer reads `sources[].id` and `sources[].root`; the `required` and `include` fields may also be retained for compatibility with source-freezing tools. Source IDs and file hashes must match `reference_results/raw_manifests/source_manifest.json`; the importer does not weaken or replace those checks.

This is the minimal complete index for `import_reference_results`. It uses only filenames already named in the public hash manifest. The directories are placeholders: place each lawfully held source at the corresponding location relative to your index, or change the relative root accordingly.

```json
{
  "schema_version": 1,
  "sources": [
    {"id": "paper_tex", "root": "manuscript/spade_dfl_v1.tex", "required": true},
    {"id": "server_archive", "root": "archives/SPADE-DFL0828.zip", "required": true},
    {"id": "brain_s0904_archive", "root": "archives/spade_dfl_brain_tumor_resnet18_pca10.zip", "required": true},
    {"id": "sensitivity_tables34", "root": "sensitivity_tables34", "required": true},
    {"id": "communication_figure10", "root": "communication_figure10", "required": true}
  ]
}
```

Author-only archive and golden-path tests can require additional historical IDs such as `sensitivity_lineage` or `sensitivity_runtime`. Add only the entries needed by the test being run, using the same index-relative rule and the unchanged public manifest identities.

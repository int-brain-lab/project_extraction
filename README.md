# project_extraction
Extraction code for IBL satellite projects e.g. U19 / personal projects

## Installation
Clone the repository, and install in place
```
pip install -e .
```

## Reference

- `projects/extractor_types.json` - (DEPRECATED) For adding custom extractors for legacy pipeline tasks
- `projects/task_extractor_map.json` - Map custom task protocols to Bpod trials extractor class
- `projects/task_type_procedures.json` - Associate Alyx procedures to a custom task protocol
- `projects/_template.py` - Example for creating a custom Bpod extractor, QC or DAQ sync task
- `projects/extraction_tasks.py` - Where to import pipeline tasks so they are readable by ibllib

## Contributing

To contribute to this repository you must create a pull request into main. Before merging you must increase the version number in the [pyproject.toml](./pyproject.toml) file (see [this guide](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers) for versioning scheme).
A GitHub check will ensure that the repository version is valid and greater than the version on the main branch. This is essential as we currently do not publish to PyPi.
The pull request may be merged only when this check passes.  Bypassing this check is not permitted, nor are direct pushes to main. Once merged, a version tag is automatically generated.

> [!IMPORTANT]
> Tests in this repository are run by both the [iblrig](https://github.com/int-brain-lab/iblrig) and [ibllib](https://github.com/int-brain-lab/ibllib) CI.

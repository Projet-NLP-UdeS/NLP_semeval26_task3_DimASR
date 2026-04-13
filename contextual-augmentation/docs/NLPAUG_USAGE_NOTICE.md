# nlpaug usage notice

This repository may include a local stub package under `nlpaug/` only to avoid
`ModuleNotFoundError` in restricted or offline environments.

For real augmentation behavior, install and use the official library:

- Project: nlpaug
- Author: Edward Ma
- GitHub: https://github.com/makcedward/nlpaug
- PyPI: https://pypi.org/project/nlpaug/

## Important

The local stub does **not** implement real contextual augmentation.
It only provides a compatible import path and a minimal API surface.

## Recommended production setup

Install official dependencies, for example:

```bash
python3 -m pip install nlpaug torch transformers sentencepiece
```

Then remove the local stub package so imports resolve to the official package.

# Contributing

## Setup

```bash
conda env create -f environment.yml
conda activate spevae
pip install -e ".[dev]"
pytest -n auto
```

The tests take about seven minutes on a laptop. The tests of the posterior
report need a trained flow checkpoint and are skipped without one.

## Reporting a problem

Open an issue. For a result that looks wrong, include the printed report and
the `.json` written next to the output: they hold everything needed to
reproduce the run except the bins.

## Changing the model or the report

A change to the model, the fit or the report is judged on known answers before
it is merged. The judges are in `scripts/step3/`, and the record of what was
measured is `docs/TRAINING.md`. A result is added to that record with the
script that produced it and a figure.

- Compare against exact answers, not against the previous checkpoint.
- Count how often the stated intervals hold the exact weight, per setting.
- Keep the negative results. They are listed in `docs/TRAINING.md` so that
  they are not tried again.

## Versions

The version is `__version__` in `src/spevae/__init__.py`. A release is a tag
`vX.Y.Z` on `main` with a section in `CHANGELOG.md`; the tag builds the
package and opens the GitHub release.

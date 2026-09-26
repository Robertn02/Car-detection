# Bike-route vehicle typology

How safe does a bike route feel? From a handlebar camera, the pipeline decides for every vehicle how it relates to
the rider - **in my lane**, **other lane, same way**, **oncoming**, **parked**, **side road / crossing** - and reports,
per ride and per second (joinable to GPS by timestamp), the exposure to each kind of traffic, the riding context
(separated path, painted bike lane, shared road), traffic signals and signs, and events such as overtakes, close
passes, stop signs and waits at red lights.

## Run it

On a machine with an NVIDIA GPU (Linux or Windows):

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m bikesafe.run videos       # folder with the ride videos
```

One command runs every stage, skips work that is already done (stages are resumable), retrains the relation model on
the labels in `data/typology/` and writes the per-ride tables to `results/corpus/`. Intermediate files go to `work/`.
To reuse finished stages from another machine, copy `work/` along with the videos; labels stay attached to the right
vehicles even when the detection stage is re-run. On USC CARC, submit `scripts/carc/run_corpus.slurm` (see the
comments at its top). `python scripts/build_upgrade_report.py` then rebuilds the report from the results.

## What is where

- `reports/UPGRADE_REPORT_2026-09.pdf` - latest report; `reports/VEHICLE_TYPOLOGY_REPORT.md` - previous milestone;
  `reports/Vehicle_Tracking_Final_Report.pdf` - first milestone (detection and tracking).
- `docs/TYPOLOGY_PIPELINE.md` - how each stage works and its known limitations.
- `bikesafe/` - the pipeline; `tests/` - unit tests (`python -m pytest tests`).
- `data/typology/` - labelled vehicles and scene frames ([labelling guide](data/typology/LABELING_GUIDE.md));
  `data/approved_*` - the hand-checked 300-frame tracking benchmark.
- `results/corpus/` - per-ride vehicle tables, per-second timelines and events; `results/typology/` - model evaluation.
- `scripts/` - report builder and tracker / vehicle-id evaluation on the benchmark.

## Privacy

The repository is public. Licence plates are blurred in every published clip and figure. Plates are read only to check
that each vehicle gets one id (`--read-plates`), and the reading is stored as a keyed hash, never as plate text.
Videos and `work/` are never committed.

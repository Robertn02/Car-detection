# Bike-route vehicle typology

How safe does a bike route feel? From a handlebar camera, the pipeline decides for every vehicle how it relates to
the rider - **in my lane**, **other lane, same way**, **oncoming**, **parked**, **side road / crossing** - and reports,
per ride and per second (joinable to GPS by timestamp), the exposure to each kind of traffic, the riding context
(separated path, painted bike lane, shared road), traffic signals, the traffic signs passed - typed and read, with the
posted speed limit - the licence plate of each vehicle, and events such as overtakes, close passes, stop signs and
waits at red lights.

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
vehicles even when the detection stage is re-run; an existing `work/` gets only what changed (the sign and lane pass
and the plate reading where they predate the October 2026 update, then the track features). The first run downloads
about 4 GB of open models (street-scene segmenter, CLIP, Florence-2) into the Hugging Face cache.

The ride videos (21 GB) do not have to sit in the project: set `BIKESAFE_VIDEOS` to their folder - for example a
Google Drive folder through Google Drive for desktop (`setx BIKESAFE_VIDEOS "G:\My Drive\bikesafe\videos"` on
Windows) - and every tool (`bikesafe.run`, `bikesafe.render`, `bikesafe.review`, ...) reads them from there; once a
ride is processed they are only read again for overlay clips and the review page.

Where to run what: on USC CARC, submit `scripts/carc_run.slurm` (one-time setup in its header) - batch jobs with no
time-outs, the best place for whole-corpus runs. On a laptop GPU the sign and lane pass is the slow stage (about 5 h
per ride on a GeForce MX250); `scripts/colab_bundle.py` runs it on a Colab GPU instead (about 22 min per ride on an
L4) through Google Drive and brings the results back - steps in its header. `python scripts/build_upgrade_report.py`
rebuilds the September report from the results.

## What is where

- `reports/UPGRADE_REPORT_2026-09.pdf` - latest report; `reports/VEHICLE_TYPOLOGY_REPORT.md` - previous milestone;
  `reports/Vehicle_Tracking_Final_Report.pdf` - first milestone (detection and tracking).
- `docs/TYPOLOGY_PIPELINE.md` - how each stage works and its known limitations; `docs/OUTPUTS.md` - every output
  table and column (start with `vehicles.csv` and `timeline_1s.csv`).
- `bikesafe/` - the pipeline; `tests/` - unit tests (`python -m pytest tests`).
- `data/typology/` - labelled vehicles and scene frames ([labelling guide](data/typology/LABELING_GUIDE.md));
  `data/approved_*` - the hand-checked 300-frame tracking benchmark.
- `results/corpus/` - per ride: `vehicles.csv` (one row per vehicle), per-second timeline, events, signs, all-feature
  track table; `results/typology/` - model evaluation.
- `scripts/` - report builder, tracker / vehicle-id evaluation on the benchmark, feature comparison and learning
  curve, and the Colab run of the sign / lane pass (`colab_bundle.py`, `colab_infrastructure.ipynb`).
- `demos/` - overlay clips of the current pipeline (relations, typed and read signs, signal states, lane lines, plates
  read): `overlay_006_bikelane_440s.mp4` (painted bike lane), `overlay_005_following_280s.mp4` (following a car from
  a red light), `overlay_006_parked_700s.mp4` (parked rows, then a wait at a red light with crossing traffic).

## Reviewing and correcting the annotations

```bash
python -m bikesafe.review           # then open http://127.0.0.1:8765
```

A browser tool for the people checking the results. It shows the pipeline's own annotations - each vehicle's relation
as a short animation of its track, each sign's type and text, each plate's reading - with the cases most likely to be
wrong first, and records every confirmation or correction (with what the model had said) in `data/typology/`.
"Retrain" rebuilds the models from the corrections. See "Human review loop" in the pipeline doc.

## Licence plates

Plates are read as plain text (`bikesafe.plates`), stored per track in `work/analysis/<video>/plates.parquet` and per
vehicle in `results/corpus/<video>/tracks_typed.csv` (`plate`), and shown in the overlay clips; nothing is blurred.
Plate text in `results/` becomes public when it is committed to this public repository. Videos and `work/` are never
committed.

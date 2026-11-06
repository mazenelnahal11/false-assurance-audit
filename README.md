# Context coverage in shortcut audits

This repository contains the experiment and verification code for a matched-budget
audit of planted visual shortcuts. The frozen run evaluates six checkpoint pairs
on 448 shared held-out cases, with separate development and calibration splits.

## Reproduce the reported tables on a CPU

Install Python 3.11 or newer and run:

```bash
python -m pip install -r requirements.txt
python audit_study.py data/frozen_study_records.zip verified_study
```

Alternatively, open `notebooks/false_assurance_results_audit.ipynb` in Colab,
choose **Run all**, and upload `data/frozen_study_records.zip` when prompted.
The audit reconstructs case-level risk labels, calibration thresholds, flags,
missed-risk rates, and paired intervals from the frozen per-case records.
The equal-review-quota diagnostic is exploratory and is labeled separately.

## Rerun the experiment on a GPU

Open `notebooks/false_assurance_study.ipynb` in Colab with a GPU and choose
**Run all**. The notebook downloads Imagenette-160 from its original source.
It trains MobileNetV3-Large and ResNet34 at seeds 11, 29, and 47, verifies development
gates, then evaluates the held-out split. The final cell exports the new results ZIP.
It leaves the model checkpoints in the runtime and records their checksums in the ZIP.

## Regenerate the illustrations

```bash
python paper_assets.py
python illustrate_study.py --tables verified_study --output figures
```

To also render the selected intervention image, supply
`--archive data/frozen_study_records.zip --dataset /path/to/imagenette2-160`
to `illustrate_study.py`. The script verifies the processed example's hash against
the archived split manifest before drawing it.

## Data contents and provenance

`data/frozen_study_records.zip` contains the original configuration, split manifest,
training assignments, development and calibration results, test responses and
decisions, query counts, and checkpoint checksums. `data/provenance.json` records
the SHA-256 values of the original export and the released records. The release
contains research-generated CSV and JSON records. The study loader obtains the
underlying images from the upstream Imagenette distribution.

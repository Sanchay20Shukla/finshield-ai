# PaySim demonstration fixture

Source: https://www.kaggle.com/datasets/ealaxi/paysim1 (Edgar Lopez-Rojas / PaySim synthetic mobile money data).

`paysim_demo.csv` contains 168 synthetic records: up to 24 per transaction-type/label group, selected deterministically by a hash of origin and step. This is an enriched UI fixture, not a random population sample or an evaluation dataset. Identifiers are synthetic. No real bank customer data is included.

The raw 493 MB source stays outside Git. `scripts/export_demo.py` documents selection and exports provenance. See the source dataset for its attribution and licensing terms; the repository does not relicense third-party data.

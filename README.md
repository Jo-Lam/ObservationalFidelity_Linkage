# Input data

Input data are not distributed with this code. Place the following files in
`data/raw/` (this folder is ignored by git).

| File | Used by | Source | SHA-256 |
|---|---|---|---|
| `ncvoter-20231002_minimal_50k.csv` | `configs/scenarios/ncvr_50k.yaml` (all 50,000 records) and `ncvr.yaml` (first 10,000) | Simple random sample of 50,000 records from the North Carolina State Board of Elections statewide voter registration snapshot of 2 October 2023 (https://www.ncsbe.gov/results-data/voter-registration-data) | `BCB54A383B4A5C4F9CBAC3CB1E1A26E7616F41BEE30CDC8273BE14BF34060E34` |
| `100000_synthetic_PDS_records_97series.xlsx` (sheet `100000_synthetic_PDS_records_97`) | `configs/scenarios/pds_100k.yaml` | Synthetic test extract with the NHS Personal Demographics Service schema, provided by NHS England | `8F80D107E4318B5798CE7CC33B6319A4D3235ADF82C65F1507961C7AE55108B4` |
| `10000_synthetic_PDS_records(in).xlsx` (sheet `10000_synthetic_PDS_records(in)`) | `configs/scenarios/pds.yaml` | Synthetic test extract with the PDS schema, provided by NHS England | `15CBBD3F57D8D93F4612EB6F1B14D41F9628165065D3AF34523F1E03491C5D66` |

Columns used: NCVR `ncid`, `first_name`, `last_name`, `ethnicity` (combined race and
ethnicity code); PDS `UNIQUE_ID`, `GIVEN_NAME`, `FAMILY_NAME`.

The PDS extracts are synthetic test data: the names are live-like in appearance but are
not names of real people. Test-harness records in the 100,000-record extract are
excluded by the `exclude` setting in `configs/scenarios/pds_100k.yaml`.

To verify a file (PowerShell: `Get-FileHash <file>`; macOS/Linux: `shasum -a 256 <file>`).

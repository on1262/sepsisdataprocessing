## Generating sepsis3.csv

The extracted sepsis3 table has **32971** rows, **25596** patients and contains the columns:
- subject_id
- stay_id
- antibiotic_time
- culture_time
- suspected_infection_time
- sofa_time
- sofa_score
- respiration
- coagulation
- liver
- cardiovascular
- cns
- renal
- sepsis3

**step1: build postgresql**

```
createdb mimiciv

cd ~/mimic-code/mimic-iv/buildmimic/postgres

psql -d mimiciv -f create.sql

psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=/path/to/mimic-iv -f load.sql

```
**step2: build concepts**

```
cd ~/mimic-code/mimic-iv/concepts_postgres

psql -d mimiciv

\i postgres-functions.sql -- only needs to be run once

\i postgres-make-concepts.sql
 
```
**step3: extract csv**

```

\copy (SELECT * FROM mimiciv_derived.sepsis3) TO '~/sepsis3.csv' WITH CSV HEADER;

```
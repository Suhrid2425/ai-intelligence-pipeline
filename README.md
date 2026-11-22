# AI Intelligence Pipeline

An end-to-end data pipeline that collects, enriches, classifies, and
resolves AI-related startup, product, research, job, and news records.

## Project overview

This project was developed as an AI engineering assignment. It combines
source-specific data collectors, evidence-based startup classification,
research-paper enrichment, entity resolution, and structured exports.

The pipeline currently produces six submission datasets:

1.  **Startups** --- 1,497 startup candidates with classification
    results and supporting evidence.
2.  **Products** --- 1,174 AI product records.
3.  **Research Papers** --- 1,000 research papers, including verified
    GitHub repository links and star counts where available.
4.  **Jobs** --- 42 AI-related job records collected from the configured
    job sources.
5.  **News** --- 7 AI news records collected from the configured news
    sources.
6.  **Entity Mapping Log** --- 2,671 startup and product
    entity-resolution records.

Counts above describe the current exported snapshot and may change when
the pipeline is run again.

## Important classification note

Startup classifications have mixed provenance:

-   Gemini: 389 records
-   Groq: 228 records
-   Local keyword rules: 880 records

The 880 local-rule results are heuristic classifications, not LLM
classifications. Their model field is `LOCAL_RULES_NOT_LLM` and their
prompt version is `local-rules-v1`. Treat these results as provisional
and review them before relying on them as verified AI classifications.

## Architecture at a glance

``` text
External sources
    |
    +--> Startup directory + startup evidence
    +--> AI product catalogue
    +--> arXiv research papers
    +--> AI news sources
    +--> AI job boards
    |
    v
Collectors and normalizers
    |
    +--> Research paper GitHub enrichment
    +--> Startup classification
    |       +--> Gemini
    |       +--> Groq
    |       +--> Local rules (fallback used for remaining records)
    |
    v
Entity resolution
    |
    v
Processed JSONL datasets
    |
    v
Excel workbook + CSV exports + audit
```

## Data sources

The current implementation uses these source families:

-   **Startups:** Y Combinator company directory data and company
    profile evidence.
-   **Products:** EveryAI tools catalogue.
-   **Research papers:** arXiv, with GitHub repository
    verification/enrichment.
-   **News:** OpenAI, Google DeepMind, Hugging Face, TechCrunch AI, and
    The Verge AI.
-   **Jobs:** Remote OK, Remotive, Arbeitnow, Jobicy, and Himalayas.

Source availability, page structure, API limits, and publication
freshness can change. Each exported record should be checked using its
source URL and collection metadata where available.

## Repository structure

``` text
ai-intelligence-pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── collectors/
│   ├── schemas/
│   ├── services/
│   ├── groq_startup_classifier.py
│   ├── export_submission.py
│   ├── audit_submission.py
│   └── classify_remaining_startups_offline.py
├── submission/
│   ├── graphone_submission.xlsx
│   └── *.csv
├── docs/
│   └── architecture.pdf
├── .env.example
└── README.md
```

The `docs/architecture.pdf` entry is the intended documentation
location; add the architecture document there when it has been created.
Do not commit `.env` or API keys.

## Setup

This project was developed with Python 3.12.

### 1. Clone the repository

``` powershell
git clone <YOUR_PUBLIC_REPOSITORY_URL>
cd ai-intelligence-pipeline
```

### 2. Create and activate a virtual environment

``` powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is present:

``` powershell
python -m pip install -r requirements.txt
```

If it is not present, create it from the working environment before
publishing the repository:

``` powershell
python -m pip freeze > requirements.txt
```

### 4. Configure API keys

Create a local `.env` file using `.env.example` as a template. Add the
API keys required by the enabled LLM providers. Never commit `.env` or
paste API keys into source code.

## Running the pipeline

Run Python modules from the repository root.

Examples:

``` powershell
python -m src.export_submission
python -m src.audit_submission
```

The collectors and enrichment modules can be run individually according
to their configuration. Review each module's arguments and
environment-variable requirements before running it.

## Submission outputs

The current export is located in `submission/`:

-   `graphone_submission.xlsx` --- workbook with six tabs.
-   CSV files --- one export per dataset.

The workbook tabs are:

-   Startups
-   Products
-   Research Papers
-   Jobs
-   News
-   Entity Mapping Log

## Validation performed

The current exported snapshot passed the project's audit checks for:

-   Required startup, product, paper, job, and news names/titles and
    source URLs.
-   Duplicate startup, product, and research-paper URLs.
-   Valid paper URLs and verified GitHub URLs where present.
-   Confidence values within the expected 0--1 range.
-   Expected six-sheet workbook structure and exported row counts.

The research-paper dataset has 142 verified GitHub links and 858 records
without a verified GitHub link. Missing GitHub links are left absent
rather than fabricated.

Entity resolution currently uses normalized exact matching and a
conservative fuzzy-match threshold. In the audited snapshot, 100
entities matched canonical seeds and 2,571 did not. Unmatched entities
are retained in the mapping log rather than being force-matched.

## Current limitations and next improvements

-   Re-run or manually review heuristic startup classifications before
    treating them as final verified labels.
-   Expand canonical seed coverage and evaluate entity-resolution
    precision/recall on a labeled validation set.
-   Add automated tests and continuous integration.
-   Add persistent storage and orchestration for larger recurring
    collection runs.
-   Implement and test additional LLM provider fallbacks and
    browser-rendering strategies before describing them as production
    capabilities.
-   Revalidate freshness windows and source coverage at each new run.

## Security

-   Keep API keys in environment variables or a local `.env` file.
-   Add `.env`, `.venv/`, caches, and temporary files to `.gitignore`.
-   Do not publish credentials, private data, or unreviewed raw files.

## License

Add the license required by the repository owner or assignment before
public release.

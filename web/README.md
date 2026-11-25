# IWDA Web Dashboard

A local browser-based interface for presenting the results of the ontology-based Web analysis project.

## Installation

Place the extracted `web` folder directly inside your existing `ontology_web_filter` project root, alongside `src`, `data`, and `scrapy.cfg`.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\web\requirements.txt
python .\web\app.py
```

Open **http://127.0.0.1:5000** in your browser. Stop with `Ctrl+C`.

## Data read by the dashboard

The app reads existing files; it does not invent results or launch a crawl:
- `data/output/classification_results_v5.json` (preferred)
- `data/output/classification_results_demo.json`
- `data/output/classification_results_v4.json`
- `data/output/classification_results.json`
- `data/ontology/ali_knowledge_graph_reorganized_v2.ont`
- `data/input/urls.txt`

The dashboard refreshes data when a page is reloaded. Keep your existing project structure unchanged.

## Pages
- Overview and classification summary
- Searchable Web page register
- Individual document details and raw JSON
- Ontology node and relationship explorer
- Methodology summary

## Access for your professor

By default, the server listens only on `127.0.0.1`, so it is accessible on the computer running it. For remote access, deploy it on an authorized server or hosting platform and configure it securely. Do not expose Flask's development server directly to the public Internet. This demo has no login/authentication.

If the dashboard does not show a field, your exported JSON may use a different field name. Update the extraction paths in `app.py` to match your actual export schema.

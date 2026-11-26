from pathlib import Path
import json
import sqlite3
import os
import sys
import subprocess
import threading
import html
import secrets
from functools import wraps
from datetime import datetime, timedelta
from collections import Counter
from flask import Flask, request, abort, render_template_string, redirect, session, url_for, send_file
from rdflib import Graph

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# Ensure the project root is importable when starting with: python web/app.py
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

OUT = PROJECT / "data" / "output"
ONTOLOGY = PROJECT / "data" / "ontology" / "ali_knowledge_graph_reorganized_v2.ont"
URLS = PROJECT / "data" / "input" / "urls.txt"
FEEDBACK_DB = OUT / "professor_feedback.sqlite3"
app = Flask(__name__)
# Set IWDA_SECRET_KEY in the hosting provider's environment variables. The
# random fallback is suitable only for local testing and invalidates sessions
# whenever the process restarts.
app.secret_key = os.environ.get("IWDA_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")),
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)
ADMIN_PASSWORD = os.environ.get("IWDA_ADMIN_PASSWORD", "")
AUTH_CONFIGURED = bool(ADMIN_PASSWORD)

# Execution is deliberately disabled by default, including on Render.
EXECUTION_ENABLED = os.environ.get("IWDA_ENABLE_EXECUTION", "0") == "1"


def init_feedback_db():
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comment TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


init_feedback_db()

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

def csrf_token():
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token

def valid_csrf():
    submitted = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    return bool(submitted and expected and secrets.compare_digest(submitted, expected))
JOB_LOCK = threading.Lock()
JOB = {
    "running": False,
    "name": None,
    "status": "IDLE",
    "output": "",
    "started_at": None,
    "finished_at": None,
    "return_code": None,
}

# Fixed allowlist: never accept arbitrary commands or shell input from the browser.
OPERATIONS = {
    "crawl": {
        "name": "Web Crawling + Complete Analysis",
        "description": "Run Scrapy, V4/V5 classification and semantic analysis.",
        "command": ["-m", "scrapy", "crawl", "ontology", "-O",
                    "data/output/classification_results_v5.json"],
        "timeout": 1800,
        "warning": "This operation overwrites classification_results_v5.json.",
    },
    "text": {
        "name": "Text Processing",
        "description": "Run the spaCy text-processing demonstration.",
        "command": ["-m", "src.text_processor"],
        "timeout": 300,
    },
    "matching": {
        "name": "Ontology Matching",
        "description": "Run ontology concept and relationship matching.",
        "command": ["-m", "src.ontology_matcher"],
        "timeout": 300,
    },
    "classification": {
        "name": "Relevance Classification",
        "description": "Run the classifier's demonstration tests.",
        "command": ["-m", "src.relevance_classifier"],
        "timeout": 300,
    },
    "semantic": {
        "name": "Semantic Analysis",
        "description": "Run semantic analysis on its demonstration document.",
        "command": ["-m", "src.semantic_analyzer"],
        "timeout": 300,
    },
    "ontology": {
        "name": "Load Ontology",
        "description": "Load the ontology and display its summary.",
        "command": ["-m", "src.ontology_loader"],
        "timeout": 300,
    },
    "extract": {
        "name": "Extract Ontology Concepts",
        "description": "Extract ontology concepts and export domain_concepts.json.",
        "command": ["-m", "src.concept_extractor"],
        "timeout": 300,
    },
}


def run_operation(operation_id):
    operation = OPERATIONS[operation_id]
    try:
        result = subprocess.run(
            [sys.executable, *operation["command"]],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=operation["timeout"],
            shell=False,
            check=False,
        )
        output = (result.stdout or "")
        if result.stderr:
            output += "\\n\\n--- STDERR ---\\n" + result.stderr
        with JOB_LOCK:
            JOB.update(
                running=False,
                status="COMPLETED" if result.returncode == 0 else "FAILED",
                output=output[-50000:],
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                return_code=result.returncode,
            )
    except subprocess.TimeoutExpired as error:
        partial = error.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        with JOB_LOCK:
            JOB.update(
                running=False,
                status="TIMEOUT",
                output=(str(partial) + "\\n\\nOperation timed out.").strip()[-50000:],
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                return_code=None,
            )
    except Exception as error:
        with JOB_LOCK:
            JOB.update(
                running=False,
                status="ERROR",
                output=str(error),
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                return_code=None,
            )

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

def load_records():
    for name in ["classification_results_v5.json", "classification_results_demo.json",
                 "classification_results_v4.json", "classification_results.json"]:
        obj = read_json(OUT / name)
        if isinstance(obj, list): return obj, name
        if isinstance(obj, dict):
            if isinstance(obj.get("items"), list): return obj["items"], name
            return [obj], name
    return [], None

def nested(obj, *paths):
    for path in paths:
        val = obj
        try:
            for key in path.split("."): val = val[key]
            if val is not None: return val
        except (KeyError, TypeError, IndexError): pass
    return None

def title(x): return str(x.get("title") or x.get("page_title") or x.get("url") or "Untitled")
def url(x): return str(x.get("url") or x.get("source_url") or "")
def status(x): return str(x.get("status") or "UNKNOWN").upper()
def cls(x):
    v = nested(x, "classification", "relevance_classification",
               "v4_analysis.classification",
               "semantic_analysis.classification.classification",
               "semantic_analysis.classification.relevance_level")
    if isinstance(v, dict): v = v.get("classification") or v.get("relevance_level")
    return str(v or "UNKNOWN").upper()
def concepts(x):
    for p in ["semantic_analysis.ontology_analysis.detected_concepts",
              "ontology_analysis.detected_concepts", "detected_concepts",
              "v4_analysis.detected_concepts"]:
        v = nested(x, p)
        if isinstance(v, list): return v
    return []
def contextual(x):
    for p in ["semantic_analysis.ontology_analysis.contextual_indicators",
              "ontology_analysis.contextual_indicators", "contextual_indicators"]:
        v = nested(x, p)
        if isinstance(v, list): return v
    return []
def data():
    records, filename = load_records()
    analyzed = [x for x in records if status(x) == "ANALYZED"]
    errors = [x for x in records if status(x) == "HTTP_ERROR"]
    counts = Counter(cls(x) for x in analyzed)
    ont = read_json(ONTOLOGY)
    nodes = ont.get("nodes", []) if isinstance(ont, dict) else []
    rels = ont.get("relations", []) if isinstance(ont, dict) else []
    try: urls = [s for s in URLS.read_text(encoding="utf-8").splitlines() if s.strip()]
    except OSError: urls = []
    return dict(records=records, filename=filename, analyzed=analyzed, errors=errors,
                counts=counts, nodes=nodes, rels=rels, urls=urls)

BASE = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ page_title }} · IWDA</title><style>
:root{--nav:#142238;--blue:#4568f5;--ink:#1b2738;--muted:#7b8798;--line:#e8edf2;--bg:#f5f7fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,sans-serif}
aside{position:fixed;inset:0 auto 0 0;width:225px;background:var(--nav);padding:28px 18px;color:white}
.brand{font-size:20px;font-weight:bold;letter-spacing:2px;margin:0 8px 38px}.brand small{display:block;color:#9eacc0;font-size:11px;letter-spacing:0;margin-top:5px}
nav a{display:block;color:#b7c4d6;text-decoration:none;padding:13px 12px;border-radius:8px;margin:4px 0}nav a:hover,nav a.active{background:#293b57;color:white}
aside footer{position:absolute;bottom:25px;color:#9eacc0;font-size:11px;line-height:1.7}
main{margin-left:225px;padding:0 38px 35px;max-width:1500px}.top{height:100px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;margin-bottom:26px}
.kicker{font-size:10px;letter-spacing:1.5px;color:#8a96a7;font-weight:bold}.top h1{font-size:25px;margin:7px 0}
h2{font-size:21px;margin:8px 0}h3{font-size:15px;margin:5px 0 17px}.muted,small{color:var(--muted)}
.hero{background:linear-gradient(110deg,#192c49,#34517f);color:white;border-radius:13px;padding:27px 30px;margin-bottom:20px}.hero p{color:#d0d9e8;max-width:700px;line-height:1.5}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}.card,.panel{background:white;border:1px solid var(--line);border-radius:11px;padding:19px;margin-bottom:17px}
.card label{font-size:10px;color:#8995a5;letter-spacing:1px;font-weight:bold}.card strong{display:block;font-size:30px;margin:10px 0 4px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.bar{height:8px;background:#edf0f4;border-radius:10px;margin:8px 0 17px;overflow:hidden}.fill{height:100%;background:#24a477}.fill.amber{background:#e3a43a}.fill.gray{background:#aeb8c5}
button,.btn{background:var(--blue);color:white;border:0;border-radius:7px;padding:10px 14px;text-decoration:none;cursor:pointer}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:13px 10px;border-bottom:1px solid #edf0f4;vertical-align:top}th{font-size:10px;color:#8995a5;letter-spacing:1px}td small{display:block;margin-top:5px;overflow-wrap:anywhere}
.badge{display:inline-block;border-radius:5px;padding:6px 8px;font-size:10px;font-weight:bold;background:#eef1f5}.direct{background:#e8f7ef;color:#16815b}.context{background:#fff4dc;color:#9c6a12}
.error{color:#bd4c42}.ok{color:#16815b}.filters{display:flex;gap:9px;margin:17px 0;flex-wrap:wrap}.filters input,.filters select{padding:10px;border:1px solid #dce3eb;border-radius:7px;background:white}
.filters input{flex:1;min-width:210px}.url{color:#738096;font-size:12px;overflow-wrap:anywhere}.item{padding:12px 0;border-bottom:1px solid #edf0f4}.item:last-child{border:0}
pre{background:#142238;color:#dbe5f2;padding:16px;border-radius:8px;white-space:pre-wrap;overflow:auto;max-height:550px;font-size:11px}
.notice{padding:14px 16px;background:#fff7e8;border:1px solid #f1e0b7;border-radius:8px;color:#80601f;line-height:1.5;margin:15px 0}
.pipeline .item{display:flex;gap:16px}.num{background:#edf2ff;color:#315efb;border-radius:8px;padding:12px;font-weight:bold;height:fit-content}
@media(max-width:850px){aside{position:static;width:auto;padding:15px}aside footer{display:none}.brand{margin:0 8px 12px}nav a{display:inline-block;padding:9px}main{margin:0;padding:0 14px}.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}}
</style></head><body><aside><div class="brand">IWDA<small>Research dashboard</small></div><nav>
<a class="{{'active' if active=='home' else ''}}" href="/">▦ &nbsp; Overview</a>
<a class="{{'active' if active=='pages' else ''}}" href="/pages">▤ &nbsp; Web pages</a>
<a class="{{'active' if active=='ontology' else ''}}" href="/ontology">◇ &nbsp; Ontology</a>
<a class="{{'active' if active=='rdf' else ''}}" href="/rdf">⌘ &nbsp; RDF Explorer</a>
<a class="{{'active' if active=='method' else ''}}" href="/method">ⓘ &nbsp; Methodology</a>
<a class="{{'active' if active=='feedback' else ''}}" href="/feedback">✎ &nbsp; Comments</a>
<a class="{{'active' if active=='execution' else ''}}" href="/execution">▶ &nbsp; Project Execution</a>
{% if session.get('authenticated') %}<a class="{{'active' if active=='feedback_management' else ''}}" href="/feedback/manage">☰ &nbsp; Feedback Management</a><form method="post" action="/logout" style="margin:8px 0 0 8px"><input type="hidden" name="csrf_token" value="{{ session.get('csrf_token','') }}"><button type="submit">Sign out</button></form>{% endif %}</nav>
<footer>LOCAL DEMONSTRATION<br>Ontology-based Web analysis</footer></aside>
<main><header class="top"><div><div class="kicker">MASTER PROJECT</div><h1>{{ heading }}</h1></div><div class="kicker">INTELLIGENT WEB DATA ANALYSIS</div></header>
{{ body|safe }}<p class="muted" style="border-top:1px solid #e5eaf0;padding-top:20px;font-size:11px">IWDA · Academic demonstration interface</p></main></body></html>
"""

def page(heading, active, body, d):
    return render_template_string(BASE, heading=heading, active=active, body=body,
                                  page_title=heading, d=d)

@app.route("/")
def home():
    d=data(); n=len(d["analyzed"]) or 1
    def bar(label,key,color):
        c=d["counts"].get(key,0)
        return f'<p>{label}<b style="float:right">{c}</b></p><div class="bar"><div class="fill {color}" style="width:{c/n*100:.1f}%"></div></div>'
    body=f"""
    <section class="hero"><div class="kicker" style="color:#b7c8e4">RESEARCH MONITOR</div><h2>Web intelligence for financial risk & fraud research</h2>
    <p>Explore collected Web documents, NLP outputs and ontology-based relevance decisions.</p><a class="btn" href="/pages">Explore documents →</a></section>
    {('<div class="notice"><b>No crawl export found.</b> Put a Scrapy JSON export in data/output/.</div>' if not d['filename'] else '')}
    <section class="cards">
    <div class="card"><label>CRAWL RECORDS</label><strong>{len(d['records'])}</strong><small>Selected JSON export</small></div>
    <div class="card"><label>ANALYZED PAGES</label><strong>{len(d['analyzed'])}</strong><small>Successfully analyzed</small></div>
    <div class="card"><label>HTTP ERRORS</label><strong>{len(d['errors'])}</strong><small>Unavailable responses</small></div>
    <div class="card"><label>INPUT URLS</label><strong>{len(d['urls'])}</strong><small>Configured URL list</small></div></section>
    <div class="grid"><section class="panel"><div class="kicker">CLASSIFICATION</div><h3>Relevance distribution</h3>
    {bar('Directly relevant','DIRECTLY_RELEVANT','')}{bar('Contextually related','CONTEXTUALLY_RELATED','amber')}{bar('Not relevant','NOT_RELEVANT','gray')}
    <small>Counts are calculated from analyzed records.</small></section>
    <section class="panel"><div class="kicker">KNOWLEDGE BASE</div><h3>Ontology status</h3>
    <div class="cards" style="grid-template-columns:repeat(3,1fr);gap:5px"><div><strong>{len(d['nodes'])}</strong><small>Nodes</small></div><div><strong>{len(d['rels'])}</strong><small>Relations</small></div><div><strong>—</strong><small>Namespaces</small></div></div><a href="/ontology">Inspect ontology →</a></section></div>
    <section class="panel"><div class="kicker">DATA SOURCE</div><h3>{d['filename'] or 'No crawl file detected'}</h3><small>Loaded from data/output. Reload this page after generating a new crawl export.</small></section>"""
    return page("Project overview","home",body,d)

@app.route("/pages")
def pages():
    d=data(); q=request.args.get("q","").lower(); filt=request.args.get("class","")
    rows=[]
    for i,x in enumerate(d["records"]):
        if q not in (title(x)+" "+url(x)).lower(): continue
        if filt and cls(x)!=filt: continue
        rows.append((i,x))
    body='<section class="panel"><p class="muted">Search and filter the records produced by the crawler.</p><form class="filters"><input name="q" placeholder="Search title or URL" value="'+q+'"><select name="class"><option value="">All classifications</option>'
    for c in ["DIRECTLY_RELEVANT","CONTEXTUALLY_RELATED","NOT_RELEVANT","UNKNOWN"]:
        body+=f'<option value="{c}" {"selected" if filt==c else ""}>{c.replace("_"," ")}</option>'
    body+='</select><button>Filter</button></form><h3>Document register · '+str(len(rows))+' record(s)</h3>'
    if rows:
        body+='<table><tr><th>DOCUMENT</th><th>CLASSIFICATION</th><th>STATUS</th><th>CONCEPTS</th><th></th></tr>'
        for i,x in rows:
            c=cls(x); badge="direct" if c=="DIRECTLY_RELEVANT" else "context" if c=="CONTEXTUALLY_RELATED" else ""
            body+=f'<tr><td><b>{title(x)}</b><small>{url(x)}</small></td><td><span class="badge {badge}">{c.replace("_"," ")}</span></td><td class="{ "ok" if status(x)=="ANALYZED" else "error"}">{status(x)}</td><td>{len(concepts(x))}</td><td><a href="/pages/{i}">Details →</a></td></tr>'
        body+='</table>'
    else: body+='<p class="muted">No records match the filters, or no crawl export is available.</p>'
    body+='</section>'
    return page("Collected Web pages","pages",body,d)

@app.route("/pages/<int:i>")
def detail(i):
    d=data()
    if i>=len(d["records"]): abort(404)
    x=d["records"][i]
    body=f'<p><a href="/pages">← Back to Web pages</a></p><section class="panel"><div class="kicker">DOCUMENT {i+1}</div><h2>{title(x)}</h2><p class="url">{url(x)}</p><span class="badge">{cls(x)}</span> &nbsp; <b>{status(x)}</b></section><div class="grid">'
    for heading,arr in [("Ontology concepts",concepts(x)),("Contextual indicators",contextual(x))]:
        body+=f'<section class="panel"><h3>{heading} ({len(arr)})</h3>'
        if arr:
            for v in arr:
                if isinstance(v,dict):
                    name=v.get("name") or v.get("label") or v.get("category") or v.get("concept_name") or "Detected item"
                    body+=f'<div class="item"><b>{name}</b><small>{v.get("id","")} {v.get("source","")}</small><p>{v.get("evidence","")}</p></div>'
                else: body+=f'<div class="item">{v}</div>'
        else: body+='<p class="muted">No data found in the corresponding fields.</p>'
        body+='</section>'
    body+='</div><section class="panel"><h3>Complete source record · JSON</h3><pre>'+json.dumps(x,ensure_ascii=False,indent=2)+'</pre></section>'
    return page("Document analysis","pages",body,d)

@app.route("/ontology")
def ontology():
    d=data(); body=f'<div class="cards"><div class="card"><label>TOTAL NODES</label><strong>{len(d["nodes"])}</strong></div><div class="card"><label>TOTAL RELATIONS</label><strong>{len(d["rels"])}</strong></div></div><div class="grid">'
    for heading,arr in [("Ontology nodes",d["nodes"]),("Ontology relationships",d["rels"])]:
        body+=f'<section class="panel"><h3>{heading} · sample</h3>'
        for x in arr[:150]:
            if isinstance(x,dict): body+=f'<div class="item"><b>{x.get("name",x.get("source",x.get("subject","Ontology entry")))}</b><small>{json.dumps(x,ensure_ascii=False)}</small></div>'
            else: body+=f'<div class="item">{x}</div>'
        body+='</section>'
    body+='</div><p class="muted">Up to 150 records of each type are displayed.</p>'
    return page("Ontology explorer","ontology",body,d)

@app.route("/method")
def method():
    d=data()
    stages=[
    ("01","Web crawling","Scrapy reads the URL list, requests accessible pages and records crawl outcomes."),
    ("02","Content extraction","HTML pages are converted into titles and textual content."),
    ("03","Text processing","spaCy performs tokenization, sentence segmentation, lemmatization, POS tagging, noun chunking and dependency parsing."),
    ("04","Concept matching","Extracted text is matched against ontology labels and aliases."),
    ("05","Contextual indicators","A separate dictionary identifies financial, cybersecurity, digital-fraud and AI expressions."),
    ("06","Semantic evidence","Explicit ontology relationships are distinguished from textual co-occurrences."),
    ("07","Relevance classification","Documents are classified as directly relevant, contextually related or not relevant."),
    ("08","Structured results","Crawl and analysis records are exported as JSON.")]
    body='<section class="hero"><h2>From Web documents to ontology-based relevance</h2><p>Implemented workflow summary for academic review.</p></section><section class="panel pipeline">'
    for n,t,desc in stages: body+=f'<div class="item"><span class="num">{n}</span><div><b>{t}</b><p class="muted">{desc}</p></div></div>'
    body+='</section><div class="notice"><b>Academic transparency:</b> the dashboard is a presentation layer. It does not itself prove that each processing stage ran successfully; inspect the JSON records and source code. The standard spaCy en_core_web_sm pipeline does not provide complete coreference resolution or full word-sense disambiguation.</div>'
    return page("Methodology","method",body,d)


@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    message = ""
    message_class = "ok"
    if request.method == "POST":
        if not valid_csrf():
            abort(400, description="Invalid security token. Reload the feedback page.")
        # Honeypot field: genuine visitors leave this hidden field empty.
        if request.form.get("website", "").strip():
            abort(400)
        comment = request.form.get("comment", "").strip()
        if not comment:
            message = "Please enter your comment before submitting."
            message_class = "error"
        elif len(comment) > 3000:
            message = "Your comment is too long. Maximum length is 3,000 characters."
            message_class = "error"
        else:
            with sqlite3.connect(FEEDBACK_DB) as conn:
                conn.execute(
                    "INSERT INTO feedback (comment, created_at) VALUES (?, ?)",
                    (comment, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")),
                )
            return redirect("/feedback?submitted=1")

    if request.args.get("submitted") == "1":
        message = "Thank you. Your comment has been submitted successfully."
        message_class = "ok"

    token = html.escape(csrf_token())
    body = f"""
    <section class="hero">
      <div class="kicker" style="color:#b7c8e4">ACADEMIC FEEDBACK</div>
      <h2>Professor Comments</h2>
      <p>Share your observations, corrections and recommendations about this research project.</p>
    </section>
    <section class="panel">
      <h3>Leave a comment</h3>
      <p class="muted">Your feedback will be saved for review by the project administrator.</p>
      <form method="post" action="/feedback">
        <input type="hidden" name="csrf_token" value="{token}">
        <div style="position:absolute;left:-10000px" aria-hidden="true">
          <label>Leave this field empty<input name="website" tabindex="-1" autocomplete="off"></label>
        </div>
        <label for="comment"><b>Your comment</b></label>
        <textarea id="comment" name="comment" rows="7" maxlength="3000" required
          placeholder="Write your comments and suggestions here..."
          style="display:block;width:100%;box-sizing:border-box;margin:10px 0 15px;padding:12px;border:1px solid #dce3eb;border-radius:7px;font:14px Arial;resize:vertical"></textarea>
        <small>Maximum 3,000 characters.</small><br><br>
        <button type="submit">Submit comment</button>
      </form>
      <p class="{message_class}">{html.escape(message)}</p>
    </section>
    """
    return page("Professor comments", "feedback", body, data())


@app.route("/feedback/manage")
@login_required
def feedback_management():
    # Only authenticated administrators can read submitted comments.
    with sqlite3.connect(FEEDBACK_DB) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        comments = conn.execute(
            "SELECT id, comment, created_at FROM feedback ORDER BY id DESC LIMIT 200"
        ).fetchall()

    body = f"""
    <section class="hero">
      <div class="kicker" style="color:#b7c8e4">ADMINISTRATOR AREA</div>
      <h2>Feedback Management</h2>
      <p>Review comments submitted by visitors and your professor.</p>
    </section>
    <section class="cards">
      <div class="card"><label>TOTAL COMMENTS</label><strong>{total}</strong><small>All submissions stored in the feedback database</small></div>
      <div class="card"><label>DISPLAYED</label><strong>{len(comments)}</strong><small>Most recent submissions (maximum 200)</small></div>
    </section>
    <section class="panel"><h3>Submitted comments</h3>
    """
    if comments:
        body += '<table><tr><th>ID</th><th>DATE (UTC)</th><th>COMMENT</th></tr>'
        for item in comments:
            body += (
                '<tr>'
                f'<td>#{int(item["id"])}</td>'
                f'<td>{html.escape(str(item["created_at"]))}</td>'
                f'<td style="white-space:pre-wrap;overflow-wrap:anywhere">{html.escape(str(item["comment"]))}</td>'
                '</tr>'
            )
        body += '</table>'
    else:
        body += '<p class="muted">No comments have been submitted yet.</p>'
    body += '</section><p><a href="/feedback">Open public comment form →</a></p>'
    return page("Feedback Management", "feedback_management", body, data())


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_CONFIGURED:
        return (
            "Authentication is not configured. Set IWDA_ADMIN_PASSWORD and "
            "IWDA_SECRET_KEY in the service environment before using this page.",
            503,
        )

    error = ""
    token = csrf_token()
    if request.method == "POST":
        if not valid_csrf():
            abort(400, description="Invalid security token. Reload the login page.")
        supplied = request.form.get("password", "")
        if secrets.compare_digest(supplied, ADMIN_PASSWORD):
            session.clear()
            session["authenticated"] = True
            session["csrf_token"] = secrets.token_urlsafe(32)
            session.permanent = True
            destination = request.args.get("next", "/execution")
            if not destination.startswith("/") or destination.startswith("//"):
                destination = "/execution"
            return redirect(destination)
        error = "Incorrect password. Please try again."
        token = csrf_token()

    return render_template_string(
        """<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">
        <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
        <title>IWDA sign in</title><style>body{font:15px Arial;background:#f5f7fa;color:#1b2738;
        display:grid;place-items:center;min-height:100vh;margin:0}.box{background:white;padding:30px;
        border:1px solid #e8edf2;border-radius:12px;width:min(360px,calc(100% - 40px))}
        input{box-sizing:border-box;width:100%;padding:12px;margin:12px 0;border:1px solid #dce3eb;
        border-radius:7px}button{background:#4568f5;color:white;border:0;border-radius:7px;
        padding:11px 16px;cursor:pointer}.error{color:#bd4c42}</style></head><body><div class=\"box\">
        <h2>IWDA · Restricted access</h2><p>Sign in to access project execution controls.</p>
        <form method=\"post\"><input type=\"hidden\" name=\"csrf_token\" value=\"{{ token }}\">
        <label for=\"password\">Access password</label><input id=\"password\" name=\"password\"
        type=\"password\" required autocomplete=\"current-password\">
        <button type=\"submit\">Sign in</button></form><p class=\"error\">{{ error }}</p>
        <p><a href=\"/\">Return to public dashboard</a></p></div></body></html>""",
        token=token, error=error,
    )


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    if not valid_csrf():
        abort(400, description="Invalid security token.")
    session.clear()
    return redirect("/")


@app.route("/execution")
@login_required
def execution():
    with JOB_LOCK:
        job = dict(JOB)

    body = """
    <section class="hero">
      <div class="kicker" style="color:#b7c8e4">CONTROLLED OPERATIONS</div>
      <h2>Run project components</h2>
      <p>Select an operation to execute a predefined project task and inspect its console output.</p>
    </section>
    """
    if not EXECUTION_ENABLED:
        body += """
        <div class="notice"><b>Execution is disabled.</b> This is the safe default for the public deployment.
        To enable the controls for a local demonstration, set the environment variable
        <code>IWDA_ENABLE_EXECUTION=1</code> before starting Flask on your own computer.</div>
        """
    else:
        body += '<div class="notice"><b>Local execution enabled.</b> Only predefined operations are available. Run the dashboard on your own computer, not on a public server.</div>'

    body += '<section class="panel"><h3>Available operations</h3><div class="grid">'
    for operation_id, operation in OPERATIONS.items():
        disabled = "" if EXECUTION_ENABLED and not job["running"] else "disabled"
        warning = operation.get("warning")
        body += (
            '<div class="card">'
            f'<b>{html.escape(operation["name"])}</b>'
            f'<p class="muted">{html.escape(operation["description"])}</p>'
            + (f'<p class="error"><small>{html.escape(warning)}</small></p>' if warning else "")
            + f'<form method="post" action="/execution/start/{operation_id}">'
            f'<input type="hidden" name="csrf_token" value="{html.escape(csrf_token())}">'
            f'<button type="submit" {disabled}>Run operation</button></form></div>'
        )
    body += '</div></section>'

    status = html.escape(str(job["status"]))
    job_name = html.escape(str(job["name"] or "No operation started"))
    output = html.escape(str(job["output"] or "No output yet."))
    started = html.escape(str(job["started_at"] or "—"))
    finished = html.escape(str(job["finished_at"] or "—"))
    return_code = html.escape(str(job["return_code"] if job["return_code"] is not None else "—"))
    body += (
        '<section class="panel"><div class="kicker">EXECUTION MONITOR</div>'
        f'<h3>{job_name}</h3><p><b>Status:</b> {status} &nbsp; '
        f'<b>Started:</b> {started} &nbsp; <b>Finished:</b> {finished} &nbsp; '
        f'<b>Exit code:</b> {return_code}</p>'
        f'<pre>{output}</pre>'
        '<p><a href="/execution">Refresh status and output</a></p></section>'
    )
    return page("Project execution", "execution", body, data())


@app.route("/execution/start/<operation_id>", methods=["POST"])
@login_required
def start_execution(operation_id):
    if not valid_csrf():
        abort(400, description="Invalid security token. Reload the execution page.")
    if not EXECUTION_ENABLED:
        abort(403)
    if operation_id not in OPERATIONS:
        abort(404)

    with JOB_LOCK:
        if JOB["running"]:
            return redirect("/execution")
        operation = OPERATIONS[operation_id]
        JOB.update(
            running=True,
            name=operation["name"],
            status="RUNNING",
            output="Operation started. Refresh this page to see the final output.",
            started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            finished_at=None,
            return_code=None,
        )

    thread = threading.Thread(
        target=run_operation,
        args=(operation_id,),
        daemon=True,
    )
    thread.start()
    return redirect("/execution")


@app.route("/rdf")
def rdf_explorer():
    try:
        from src.rdf_exporter import export as export_rdf
        ttl_path, rdf_path, triple_count = export_rdf()
        graph = Graph()
        graph.parse(str(ttl_path), format="turtle")
        page_num = max(1, request.args.get("page", 1, type=int))
        page_size = 50
        triples = sorted(graph, key=lambda t: (str(t[0]), str(t[1]), str(t[2])))
        total_pages = max(1, (len(triples) + page_size - 1) // page_size)
        page_num = min(page_num, total_pages)
        visible = triples[(page_num - 1) * page_size:page_num * page_size]
        body = f'''<section class="hero"><div class="kicker" style="color:#b7c8e4">SEMANTIC WEB EXPORT</div>
        <h2>RDF Explorer</h2><p>Browse the RDF graph generated from the ontology and the latest crawl analysis.</p></section>
        <section class="cards"><div class="card"><label>TOTAL RDF TRIPLES</label><strong>{triple_count}</strong><small>Unique subject-predicate-object statements</small></div>
        <div class="card"><label>EXPORT FORMAT</label><strong>TTL</strong><small>Turtle serialization</small></div>
        <div class="card"><label>EXPORT FORMAT</label><strong>RDF/XML</strong><small>XML serialization</small></div>
        <div class="card"><label>PAGE</label><strong>{page_num}/{total_pages}</strong><small>{len(visible)} triples displayed</small></div></section>
        <section class="panel"><h3>Download RDF exports</h3><p><a class="btn" href="/rdf/download/ttl">Download .ttl</a> &nbsp; <a class="btn" href="/rdf/download/rdf">Download .rdf</a></p>
        <small>Exports are regenerated from data/output/classification_results_v5.json and the project ontology.</small></section>
        <section class="panel"><h3>Triple register</h3><div style="overflow-x:auto"><table><tr><th>SUBJECT</th><th>PREDICATE</th><th>OBJECT</th></tr>'''
        for subject, predicate, obj in visible:
            body += '<tr>' + ''.join(f'<td style="overflow-wrap:anywhere">{html.escape(str(value))}</td>' for value in (subject, predicate, obj)) + '</tr>'
        body += '</table></div><p>'
        if page_num > 1:
            body += f'<a href="/rdf?page={page_num-1}">Previous</a> &nbsp; '
        if page_num < total_pages:
            body += f'<a href="/rdf?page={page_num+1}">Next</a>'
        body += '</p></section><div class="notice"><b>Interpretation:</b> ontology assertions are exported as graph relations. Document-to-concept links represent detected concept mentions. Textual co-occurrences are stored separately as evidence and are not asserted as ontology relations.</div>'
        return page("RDF Explorer", "rdf", body, data())
    except Exception as error:
        message = html.escape(str(error))
        body = f'<section class="panel"><h3>RDF export unavailable</h3><p class="error">{message}</p><p>Check that rdflib is installed and the crawl JSON and ontology files exist in the expected project folders.</p></section>'
        return page("RDF Explorer", "rdf", body, data()), 500


@app.route("/rdf/download/<fmt>")
def rdf_download(fmt):
    if fmt not in ("ttl", "rdf"):
        abort(404)
    from src.rdf_exporter import export as export_rdf
    ttl_path, rdf_path, _ = export_rdf()
    path = ttl_path if fmt == "ttl" else rdf_path
    mimetype = "text/turtle" if fmt == "ttl" else "application/rdf+xml"
    return send_file(path, as_attachment=True, download_name=path.name, mimetype=mimetype)


if __name__=="__main__":
    print("IWDA dashboard: http://127.0.0.1:5000")
    app.run(host="127.0.0.1",port=5000,debug=False)

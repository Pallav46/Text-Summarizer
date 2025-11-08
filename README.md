# Professional Text Summarization (Flask)

A small, self-hosted Flask web app that performs text summarization (abstractive and extractive) and can fetch YouTube transcripts for summarization. The app is intentionally defensive: heavy dependencies (PyTorch / Transformers / spaCy) are loaded lazily and the server falls back to a lightweight extractive approach when those libraries or models are unavailable.

## Highlights
- Web UI (root `/`) for pasting a YouTube link and generating either an abstractive or extractive summary.
- REST API endpoint `/summarize` to integrate summarization into other tools or scripts.
- Uses `youtube_transcript_api` (via the project's `temp.py`) to fetch and format transcripts.
- Abstractive summarization via the Hugging Face `transformers` pipeline when available; otherwise falls back to a spaCy-based or simple frequency-based extractive summarizer.

## Project layout

- `app.py` — Flask application and summarization logic (abstractive + extractive, transcript integration).
- `temp.py` — YouTube transcript helper (fetch, convert segments to text, optional translation).
- `templates/index.html` — Simple web UI for entering a YouTube URL and getting a summary.
- `test_extractive.py` — A small runtime test for the extractive summarizer.
- `requirements.txt` — Top-level runtime dependencies.
 - `temp.py` — YouTube transcript helper (fetch, convert segments to text, optional translation).
 - `templates/index.html` — Simple web UI for entering a YouTube URL and getting a summary.
 - `requirements.txt` — Top-level runtime dependencies.

## Quick start (Windows PowerShell)

Recommended: create and use a virtual environment before installing dependencies.

1. Create & activate a venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. (Optional) Upgrade pip and install developer tools:

```powershell
python -m pip install -U pip setuptools wheel
python -m pip install pip-tools
```

3. Install dependencies (see "Dependency pinning" below for making this deterministic):

```powershell
pip install -r .\requirements.txt
```

4. Run the Flask app locally (development):

```powershell
# from repository root
$env:FLASK_APP = 'app.py'
python app.py
# or
# flask run --host=0.0.0.0 --port=5000
```

Open http://127.0.0.1:5000 in a browser to use the UI.

## API — /summarize

POST /summarize (application/json)

Request payload (examples):

Abstractive (attempts abstractive; falls back to extractive on failure):

```json
{
	"type": "abstractive",
	"youtube_url": "https://www.youtube.com/watch?v=<VIDEO_ID>"
}
```

Extractive with a requested number of sentences:

```json
{
	"type": "extractive",
	"youtube_url": "https://www.youtube.com/watch?v=<VIDEO_ID>",
	"num_sentences": 3
}
```

Response (JSON):

```json
{
	"summary": "...",
	"original_length": 1234,
	"summary_length": 123,
	"compression_rate": "90.1%",
	"summary_points": ["Sentence 1","Sentence 2"]
}
```

Errors are returned as JSON with an `error` field and appropriate HTTP status codes.

## Notes on behavior and fallback

- Abstractive summarization depends on large ML models (Transformers & PyTorch). If those packages/models are missing or fail to load, the server will automatically fall back to extractive summarization so the service remains functional.
- The Flask app is written to attempt an automatic download of `en_core_web_sm` if spaCy is not present; automatic downloads may fail on locked-down systems — in that case install `spacy` and the model manually:

```powershell
pip install spacy
python -m spacy download en_core_web_sm
```

## Dependency pinning / permanent fix for slow pip resolution

If you saw messages like "pip is looking at multiple versions of google-api-core[grpc]..." or long resolver delays, the root cause is unpinned or loosely pinned dependencies which cause pip's resolver to explore many combinations.

Recommended approaches to make installs fast and deterministic:

1. Use a constraints file (pin troublesome transitive dependencies) and reference it in `pip install`:

```text
# constraints.txt (example)
google-auth-httplib2==0.2.0
google-api-core==2.28.0
googleapis-common-protos==1.56.2
# add other pinned transitive deps as needed
```

Then install with:

```powershell
pip install -r requirements.txt -c constraints.txt
```

2. Use `pip-tools` (`pip-compile`) to produce a fully pinned `requirements.txt` from a simple `requirements.in` (recommended for projects you deploy or share):

```powershell
# create requirements.in with top-level deps (no transitive pins)
pip-compile requirements.in --output-file=requirements.txt
pip install -r requirements.txt
```

This ensures the hard work of resolving combinations happens once (when you compile), not each time someone runs `pip install`.

3. Upgrade pip (modern pip versions perform better); enable `--use-feature=fast-deps` only if recommended by your pip version's docs (pip's features change across versions).

If you want, I can produce a suggested `constraints.txt` that pins the Google client libraries observed in resolver messages — tell me and I'll add it to the repo.

## Troubleshooting

- Long resolver times: pin transitive dependencies (see above).
- `youtube_transcript_api` failures: ensure network access and that transcripts are available for the requested video. Some videos have transcripts disabled.
- Abstractive model memory errors: choose a smaller model (e.g., `t5-small`) or run on a machine with a GPU and sufficient VRAM.

## Tests

There are no project-specific tests included by default. If you add automated tests (for example, using pytest), include a `tests/` directory and a CI workflow to run them.

If you need a quick manual check of the extractor behavior, you can run a short Python snippet that imports `extractive_summarize` from `app.py` and prints the output for a sample paragraph.

## Contributing

Contributions are welcome. Suggested small, safe changes:

- Add a CI job that pins dependencies and runs the test.
- Add Dockerfile or GitHub Actions workflow to run the app inside a reproducible container.

## License

No license included. Add a `LICENSE` file (for example, MIT) if you intend to make this code permissively available.

---
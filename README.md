# CrewAI Projects API

FastAPI backend for six CrewAI projects adapted from the supplied Google Colab notebook.

## Projects and routes

| Route | Project | Input expected |
|---|---|---|
| `POST /agent1` | Blog writing | Blog topic |
| `POST /agent2` | Competitor analysis | Competitor names and/or website URLs |
| `POST /agent3` | Meeting summarizer | Meeting transcript text |
| `POST /agent4` | Sales data analysis | CSV file content pasted as text |
| `POST /agent5` | Customer support | Customer support ticket |
| `POST /agent6` | Approval-based report | Topic and optional audience/decision fields |

All routes accept JSON in this format:

```json
{
  "input": "Your text here"
}
```

All routes return:

```json
{
  "output": "Generated result"
}
```

## 1. Install Python packages

```bash
pip install -r requirements.txt
```

## 2. Set the Gemini API key

Set an environment variable named `GEMINI_API_KEY`. Do not paste the key into `app.py`, GitHub, or this README.

Linux/macOS:

```bash
export GEMINI_API_KEY="your-key"
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your-key"
```

On a hosting platform, add `GEMINI_API_KEY` in the service's Environment Variables settings.

## 3. Run the API locally

```bash
uvicorn app:app --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

## 4. Test a route

In Postman:
- Method: `POST`
- URL: `http://127.0.0.1:8000/agent1`
- Body: raw → JSON

```json
{
  "input": "The uses of artificial intelligence in education"
}
```

## Notes

- This backend contains the six projects present in the supplied notebook.
- The sales analysis route expects CSV contents as text in the `input` field.
- For `/agent6`, use text such as:

```text
Topic: Artificial intelligence in healthcare
Audience: College students
Decision: approve
```

- If `Decision` is omitted, the endpoint returns a draft. Use `Decision: approve` or `Decision: reject` to record the requested decision/revision path.
- The API key must be configured on the machine or hosting service running this backend.
- CrewAI tasks can take time because they call an external language model and, for some projects, web search.

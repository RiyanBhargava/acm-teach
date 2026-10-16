"""Optional demo interface. Run this file; teach the five core modules."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dotenv import load_dotenv

from graph import app
from main import ROOT
from nodes import OUTPUT_FILE


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI App Builder</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#10151d;color:#edf2f7;font:15px system-ui,sans-serif}
header{padding:24px 30px;border-bottom:1px solid #303947}h1{margin:5px 0;font-size:26px}
.eyebrow{color:#91e3c5;font-size:12px;letter-spacing:2px}p{color:#aab8c8;margin:8px 0}
form{display:flex;gap:12px;padding:22px 30px}textarea{flex:1;resize:vertical;min-height:75px;background:#1b2430;color:inherit;border:1px solid #465268;border-radius:8px;padding:14px;font:inherit}
button{border:0;border-radius:7px;padding:12px 20px;background:#a5efd4;color:#112820;font:600 14px system-ui;cursor:pointer}button:disabled{opacity:.5;cursor:wait}
#status{margin:0 30px 20px;min-height:22px}main{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:0 30px 30px}
section{min-width:0;border:1px solid #303947;border-radius:10px;overflow:hidden;background:#17202b}
.bar{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid #303947;gap:12px}
h2{font-size:14px;margin:0}.bar button{padding:7px 12px;background:#29384a;color:#edf2f7}
pre{height:60vh;overflow:auto;margin:0;padding:20px;font:13px/1.65 Consolas,monospace;tab-size:4;white-space:pre}
iframe{border:0;width:100%;height:60vh;background:white;display:block}a{color:#a5efd4;font-size:13px}
@media(max-width:800px){main{grid-template-columns:1fr}form{flex-direction:column}header,form{padding:20px}main{padding:0 20px 20px}}
</style></head><body>
<header><div class="eyebrow">IDEA → CODE → WEBSITE</div><h1>AI App Builder</h1>
<p>Describe a small app. Explore the code and try the result.</p></header>
<form id="form"><textarea id="prompt" aria-label="App description" required maxlength="10000" placeholder="Build a counter with Increase, Decrease, and Reset buttons."></textarea><button id="generate">Generate app</button></form>
<p id="status" role="status" aria-live="polite">Ready to build.</p>
<main><section><div class="bar"><h2>Files / index.html</h2><a id="download" hidden download="index.html">Download HTML</a></div><pre><code id="code">Your generated code will appear here.</code></pre></section>
<section><div class="bar"><h2>Website preview</h2><button id="refresh" disabled>Restart preview</button></div><iframe id="preview" title="Generated website preview" sandbox="allow-scripts" referrerpolicy="no-referrer"></iframe></section></main>
<script>
const form=document.querySelector('#form'),button=document.querySelector('#generate');
const status=document.querySelector('#status'),code=document.querySelector('#code');
const preview=document.querySelector('#preview'),refresh=document.querySelector('#refresh');
const download=document.querySelector('#download');let html='',downloadUrl;
// The preview runs scripts in an isolated frame; it cannot access this interface.
function render(){
 const policy="<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'\">";
 preview.srcdoc=policy+html;
}
function show(content){html=content;code.textContent=html;render();refresh.disabled=false;
 if(downloadUrl)URL.revokeObjectURL(downloadUrl);
 downloadUrl=URL.createObjectURL(new Blob([html],{type:'text/html'}));download.href=downloadUrl;download.hidden=false;
}
refresh.onclick=render;
form.onsubmit=async(event)=>{event.preventDefault();const prompt=document.querySelector('#prompt').value.trim();if(!prompt)return;
 button.disabled=true;status.textContent='Generating your app…';
 try{const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt})});
 const data=await response.json();if(!response.ok)throw new Error(data.error);show(data.html);status.textContent='Saved output/index.html. Try the app in the preview.';
 }catch(error){status.textContent='Generation failed: '+error.message+(html?' Previous result is still displayed.':'');}
 finally{button.disabled=false;}
};
fetch('/api/file').then(r=>r.json()).then(data=>{if(data.html){show(data.html);status.textContent='Loaded the last saved app.';}}).catch(()=>{status.textContent='Could not load the saved app. You can still generate a new one.';});
</script></body></html>'''

GENERATION_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def respond(self, data, status=200, content_type="application/json"):
        body = json.dumps(data).encode() if content_type == "application/json" else data.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.respond(PAGE, content_type="text/html")
        elif self.path == "/api/file":
            with GENERATION_LOCK:
                html = OUTPUT_FILE.read_text(encoding="utf-8") if OUTPUT_FILE.exists() else ""
            self.respond({"html": html})
        else:
            self.respond({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/api/generate":
            self.respond({"error": "Not found"}, 404)
            return
        # Accept requests from this local UI, not forms on other websites.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://127.0.0.1:{self.server.server_port}":
            self.respond({"error": "Origin not allowed"}, 403)
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.respond({"error": "Expected JSON"}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 50000:
                raise ValueError("Request is empty or too large")
            data = json.loads(self.rfile.read(length))
            prompt = data.get("prompt") if isinstance(data, dict) else None
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10000:
                raise ValueError("Enter an app description of 1–10000 characters")
        except (ValueError, UnicodeDecodeError) as error:
            self.respond({"error": str(error)}, 400)
            return
        if not os.getenv("GROQ_API_KEY"):
            self.respond({"error": "Add GROQ_API_KEY to this project's .env file and restart."}, 503)
            return
        if not GENERATION_LOCK.acquire(blocking=False):
            self.respond({"error": "Another generation is running. Try again shortly."}, 409)
            return
        try:
            result = app.invoke({"request": prompt.strip()})
            self.respond({"html": result["html"]})
        except Exception:
            self.respond({"error": "The build failed. Check your model access, connection, or try a smaller request."}, 502)
        finally:
            GENERATION_LOCK.release()


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("AI App Builder: http://127.0.0.1:8765 — press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

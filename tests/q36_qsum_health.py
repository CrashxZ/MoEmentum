#!/usr/bin/env python3
import ssl,json,urllib.request,time,subprocess
from pathlib import Path
base='https://192.168.8.209:8443';ctx=ssl.create_default_context(cafile='/etc/llama-inference/server.crt');key=next(x.strip() for x in Path('/etc/llama-inference/api-keys').read_text().splitlines() if x.strip() and not x.startswith('#'))
def req(path,payload=None):
 r=urllib.request.Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 with urllib.request.urlopen(r,context=ctx,timeout=150) as f:return json.load(f)
models=req('/v1/models');model=models['data'][0]['id'];response=req('/v1/chat/completions',dict(model=model,messages=[dict(role='user',content='Reply with OK.')],max_tokens=2,temperature=0,stream=False))
print(json.dumps(dict(service=subprocess.check_output(['systemctl','is-active','llama-inference.service'],text=True).strip(),enabled=subprocess.check_output(['systemctl','is-enabled','llama-inference.service'],text=True).strip(),tls_verified=True,authenticated=True,model=model,completion_received=bool(response.get('choices')),usage=response.get('usage'),secret_printed=False),indent=2))

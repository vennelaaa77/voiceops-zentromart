#!/usr/bin/env python
"""Voice demo page (for the 2-minute video):  python demo_app.py  ->  http://127.0.0.1:5000
Hold the mic button, speak, release. Shows transcript, spoken reply, sources and tool calls."""
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from src.agent import SupportAgent
from src.config import ROOT
from src.kb import KnowledgeBase
from src.llm import LLM
from src.voice import Voice

app = Flask(__name__)
OUT = ROOT / "outputs" / "demo"; OUT.mkdir(parents=True, exist_ok=True)
voice, llm, kb = Voice(), LLM(), KnowledgeBase()
agent = SupportAgent(llm, kb, mode="rag", channel="voice")

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ZentroMart VoiceOps</title>
<style>
body{font:16px system-ui;max-width:640px;margin:2rem auto;padding:0 1rem;background:#fcf8f4;color:#232b36}
h1{font-size:1.4rem}#log{min-height:260px}.b{padding:.6rem .9rem;margin:.5rem 0;border-radius:12px;max-width:85%}
.u{background:#e7e3f7;margin-left:auto}.a{background:#fff;border:1px solid #e6dfd6}.m{font-size:.75rem;color:#6a7280}
button{font:inherit;padding:.8rem 1.2rem;border-radius:999px;border:0;background:#232b36;color:#fff;cursor:pointer}
button.rec{background:#d1452f}input{font:inherit;padding:.6rem;width:60%}
</style>
<h1>ZentroMart support &middot; voice</h1><div id=log></div>
<p><button id=mic>Hold to talk</button> <button id=reset style="background:#8a93a3">New call</button></p>
<p><input id=t placeholder="or type a message"> <button id=send>Send</button></p>
<script>
const log=document.getElementById('log');let ctx,stream,proc,chunks=[],rate;
function add(c,t,m){const d=document.createElement('div');d.className='b '+c;d.textContent=t;if(m){const s=document.createElement('div');s.className='m';s.textContent=m;d.appendChild(s)}log.appendChild(d);d.scrollIntoView()}
function wav(f32,inRate){const r=16000,ratio=inRate/r,n=Math.floor(f32.length/ratio),b=new DataView(new ArrayBuffer(44+n*2));
const w=(o,s)=>{for(let i=0;i<s.length;i++)b.setUint8(o+i,s.charCodeAt(i))};w(0,'RIFF');b.setUint32(4,36+n*2,true);w(8,'WAVEfmt ');
b.setUint32(16,16,true);b.setUint16(20,1,true);b.setUint16(22,1,true);b.setUint32(24,r,true);b.setUint32(28,r*2,true);b.setUint16(32,2,true);b.setUint16(34,16,true);
w(36,'data');b.setUint32(40,n*2,true);for(let i=0;i<n;i++){const x=Math.max(-1,Math.min(1,f32[Math.floor(i*ratio)]));b.setInt16(44+i*2,x<0?x*32768:x*32767,true)}return new Blob([b],{type:'audio/wav'})}
async function start(){stream=await navigator.mediaDevices.getUserMedia({audio:true});ctx=new AudioContext();rate=ctx.sampleRate;chunks=[];
const src=ctx.createMediaStreamSource(stream);proc=ctx.createScriptProcessor(4096,1,1);proc.onaudioprocess=e=>chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
src.connect(proc);proc.connect(ctx.destination);mic.classList.add('rec');mic.textContent='Listening...'}
async function stop(){if(!proc)return;proc.disconnect();stream.getTracks().forEach(t=>t.stop());mic.classList.remove('rec');mic.textContent='Hold to talk';
const len=chunks.reduce((a,c)=>a+c.length,0),all=new Float32Array(len);let o=0;chunks.forEach(c=>{all.set(c,o);o+=c.length});proc=null;
const fd=new FormData();fd.append('audio',wav(all,rate),'turn.wav');turn(fd)}
async function turn(body,isJson){add('m','...');const r=await fetch('/api/turn',isJson?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{method:'POST',body});
const j=await r.json();log.lastChild.remove();add('u',j.transcript||'(nothing heard)');add('a',j.reply,'sources: '+(j.cited.join(', ')||j.retrieved.slice(0,4).join(', '))+(j.tools.length?'  |  tools: '+j.tools.join(', '):''));
if(j.audio){new Audio(j.audio+'?'+Date.now()).play()}}
mic.onmousedown=start;mic.onmouseup=stop;mic.ontouchstart=e=>{e.preventDefault();start()};mic.ontouchend=stop;
send.onclick=()=>{if(t.value.trim()){turn({text:t.value},true);t.value=''}};t.onkeydown=e=>{if(e.key==='Enter')send.click()};
reset.onclick=async()=>{await fetch('/api/reset',{method:'POST'});log.innerHTML=''};
</script>"""


@app.get("/")
def index():
    return PAGE


@app.post("/api/reset")
def reset():
    agent.reset()
    return jsonify(ok=True)


@app.post("/api/turn")
def turn():
    if "audio" in request.files:
        f = OUT / "in.wav"; request.files["audio"].save(f)
        heard = voice.transcribe(f)
    else:
        heard = (request.get_json() or {}).get("text", "")
    if not heard.strip():
        reply, res = "Sorry, I didn't catch that. Could you say it again?", {"used": [], "retrieved": [], "tool_calls": []}
    else:
        res = agent.respond(heard); reply = res["reply"]
    name = f"reply_{int(time.time()*1000)}.wav"
    voice.speak(reply, OUT / name)
    return jsonify(transcript=heard, reply=reply, cited=res["used"], retrieved=res["retrieved"],
                   tools=[c["name"] for c in res["tool_calls"]],
                   audio=f"/audio/{name}" if (OUT / name).exists() else None)


@app.get("/audio/<name>")
def audio(name):
    return send_file(OUT / Path(name).name, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

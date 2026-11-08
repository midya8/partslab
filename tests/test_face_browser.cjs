'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),http=require('node:http'),{spawn}=require('node:child_process');
const {waitForDebugger}=require('./chrome_support.js');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
test('real Chrome:54 exact face workers, endpoint queries, source binding, stale contexts, exports and mobile', {timeout:180000}, async()=>{
 const root=path.resolve(__dirname,'../docs');const server=http.createServer((req,res)=>{const file=path.join(root,new URL(req.url,'http://localhost').pathname);if(!file.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}fs.readFile(file,(err,data)=>{if(err){res.writeHead(404);res.end();return;}res.setHeader('Content-Type',file.endsWith('.js')?'application/javascript':file.endsWith('.css')?'text/css':'text/html');res.end(data);});});await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const target=process.env.PARTSLAB_FACE_URL||`http://127.0.0.1:${server.address().port}/face.html`;
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'parts-face-browser-'));
 const browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--disable-dev-shm-usage','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});let ws,startupError,diagnostics='';browser.on('error',e=>startupError=e);browser.stderr.on('data',d=>diagnostics=(diagnostics+d).slice(-4000));
 try{
  for(let i=0;i<1200&&!startupError&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);
  assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),`Chrome debugger did not start: ${startupError?.message||diagnostics||browser.exitCode}`);const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];
  const tab=await waitForDebugger(`http://127.0.0.1:${port}/json/list`,{alive:()=>browser.exitCode===null&&!startupError});ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});
  let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {res,rej}=pending.get(m.id);pending.delete(m.id);m.error?rej(new Error(JSON.stringify(m.error))):res(m.result);}};
  const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));});
  const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};

  await call('Page.enable');await call('Page.navigate',{url:target});
  async function ready(){for(let i=0;i<1500;i++){const s=await js(`document.getElementById('status')?.textContent||''`);if(s.startsWith('Exact face verified'))return;if(s==='Worker error.'||s==='Refused without repair.')throw Error(await js(`document.getElementById('error').textContent`));await delay(20);}throw Error('face worker timeout');}
  await ready();
  const manifest=JSON.parse(fs.readFileSync(path.join(__dirname,'../partslab/artifacts/face-reference.json'),'utf8')),native={...manifest,cases:manifest.shards.flatMap(name=>JSON.parse(fs.readFileSync(path.join(__dirname,'../partslab/artifacts',name),'utf8')).cases)};
  for(const c of native.cases){
   await js(`document.getElementById('problem').value='source';document.getElementById('problem').dispatchEvent(new Event('change'));document.getElementById('rank').value='${c.rank}';document.getElementById('run').value='${c.run}';document.getElementById('row').value='${c.row}';document.getElementById('pixel').value='${c.pixel}';document.getElementById('apply').click()`);await ready();
   assert.deepEqual(await js(`PARTS_FACE_STATE.witness`),c.witness);
   for(const endpoint of ['lower','zero','upper']){
    await js(`document.getElementById('${endpoint}').click()`);
    const state=await js(`PARTS_FACE_STATE`),w=c.witness;
    const read=r=>[BigInt(r.numerator),BigInt(r.denominator)];const [tn,td]=read(state.delta);
    w.coefficients.forEach((v,k)=>{const [an,ad]=read(v.intercept),[bn,bd]=read(v.slope),[cn,cd]=read(state.coefficients[k]);assert.equal(cn*ad*bd*td,(an*bd*td+bn*ad*tn)*cd);assert(cn>=0n);});
    assert.match(await js(`document.getElementById('query-status').textContent`),/All exact KKT signs pass/);
    assert.equal(await js(`(()=>{const s=PARTS_FACE_STATE,w=s.witness,t=PartsFaceUI.approximate(PartsFace.read(s.delta)),ctx=document.getElementById('digit').getContext('2d');for(let j=0;j<64;j++){const expected=Math.round((w.problem.pixels[j]+(j===w.problem.pixel?t:0))*255),rgba=ctx.getImageData(j%8*32+16,Math.floor(j/8)*32+16,1,1).data;if(rgba[0]!==expected||rgba[1]!==expected||rgba[2]!==expected)return false;}return true;})()`),true);

   }
  }
  for(let i=0;i<4;i++){
   await js(`document.getElementById('problem').value='${i}';document.getElementById('problem').dispatchEvent(new Event('change'));document.getElementById('apply').click()`);await ready();assert.deepEqual(await js(`PARTS_FACE_STATE.witness`),native.controls[i]);
  }
  const prior=await js(`JSON.stringify(PARTS_FACE_STATE)`);
  await js(`document.getElementById('delta').value='1';document.getElementById('query').click()`);assert.match(await js(`document.getElementById('query-error').textContent`),/Outside/);assert.equal(await js(`JSON.stringify(PARTS_FACE_STATE)`),prior);
  for(const value of ['1/0','1e-5','oops','--1']){await js(`document.getElementById('delta').value=${JSON.stringify(value)};document.getElementById('query').click()`);assert.equal(await js(`JSON.stringify(PARTS_FACE_STATE)`),prior);}
  await js(`document.getElementById('delta').value='.125';document.getElementById('delta').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}))`);assert.deepEqual(await js(`PARTS_FACE_STATE.delta`),{numerator:'1',denominator:'8'});
  assert.equal(await js(`(()=>{const v=document.getElementById('digit').getContext('2d').getImageData(64,64,1,1).data;return v[0]===159&&v[1]===159&&v[2]===159;})()`),true);
  const saved=await js(`JSON.stringify(PARTS_FACE_STATE.witness)`);await js(`document.getElementById('problem').value='source';document.getElementById('problem').dispatchEvent(new Event('change'));document.getElementById('pixel').value='1.5';document.getElementById('apply').click()`);assert.equal(await js(`JSON.stringify(PARTS_FACE_STATE.witness)`),saved);assert.match(await js(`document.getElementById('error').textContent`),/integer pixel/);
  await js(`document.getElementById('pixel').value='27';document.getElementById('rank').value='32';document.getElementById('apply').click();document.getElementById('problem').value='1';document.getElementById('apply').click()`);await ready();assert.equal(await js(`PARTS_FACE_STATE.control`),1);await delay(50);assert.equal(await js(`PARTS_FACE_STATE.control`),1);assert.match(await js(`document.getElementById('bounds').textContent`),/singleton/);
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('denied')}},configurable:true});document.getElementById('copy').click()`);await delay(20);assert.match(await js(`document.getElementById('export-status').textContent`),/Copy blocked/);
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async text=>globalThis.copied=text},configurable:true});document.getElementById('copy').click()`);await delay(20);assert.deepEqual(JSON.parse(await js(`copied`)),await js(`PARTS_FACE_STATE.witness`));
  await js(`URL.createObjectURL=(blob)=>{globalThis.exportBlob=blob;return 'blob:face-test'};HTMLAnchorElement.prototype.click=function(){globalThis.downloaded=this.download};document.getElementById('download').click()`);assert.equal(await js(`downloaded`),'partslab-face-witness.json');assert.deepEqual(JSON.parse(await js(`exportBlob.text()`)),await js(`PARTS_FACE_STATE.witness`));
  await js(`for(const e of document.querySelectorAll('details'))e.open=true;document.getElementById('problem').value='source';document.getElementById('problem').dispatchEvent(new Event('change'));document.getElementById('rank').value='16';document.getElementById('row').value='0';document.getElementById('run').value='0';document.getElementById('pixel').value='27';document.getElementById('apply').click()`);await ready();
  for(const width of [320,390,1280]){await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});await delay(30);assert.ok(await js(`document.documentElement.scrollWidth<=innerWidth`),'overflow at '+width);assert.ok(await js(`Array.from(document.querySelectorAll('button,select,summary,input')).filter(e=>e.getBoundingClientRect().height>0).every(e=>e.getBoundingClientRect().height>=44)`),'touch at '+width);}
  if(process.env.PARTSLAB_FACE_SCREENSHOT){await js(`for(const e of document.querySelectorAll('details'))e.open=false;scrollTo(0,0)`);const width=Number(process.env.PARTSLAB_FACE_WIDTH||1280);await call('Emulation.setDeviceMetricsOverride',{width,height:1200,deviceScaleFactor:1,mobile:width<700});const metrics=await call('Page.getLayoutMetrics');const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:metrics.cssContentSize.height,scale:1}});fs.writeFileSync(process.env.PARTSLAB_FACE_SCREENSHOT,Buffer.from(shot.data,'base64'));}
 }finally{if(ws)ws.close();if(browser.pid&&browser.exitCode===null){browser.kill('SIGTERM');await new Promise(resolve=>browser.once('exit',resolve));}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});await new Promise(r=>server.close(r));}
});

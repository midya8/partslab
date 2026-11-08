'use strict';
(()=>{
 const $=id=>document.getElementById(id),F=PartsFace;let worker=null,sequence=0,applied=null,lastQuery=null;
 const controls=['Unused gradient binds','Singleton empty face','Duplicate components','Domain / coefficient tie'];
 for(let i=0;i<20;i++)$('row').add(new Option(String(i),String(i)));
 const approximate=r=>{const n=r[0],d=r[1];if(n===0n)return 0;const a=String(n<0n?-n:n),b=String(d);return (n<0n?-1:1)*Number(a.slice(0,15))/Number(b.slice(0,15))*10**(Math.max(0,a.length-15)-Math.max(0,b.length-15));};
 const text=r=>{const v=approximate(r);return v===0?'0':Number.isFinite(v)?v.toPrecision(7):'outside display range';};
 function parse(input){
  const s=input.trim();if(!s||s.length>100)throw Error('Use a decimal or integer fraction, at most 100 characters.');
  const m=s.match(/^([+-]?\d+)\/(\d+)$/);if(m){const d=BigInt(m[2]);if(d===0n)throw Error('Denominator must be positive.');return F.add([BigInt(m[1]),d],[0n,1n]);}
  if(!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(s))throw Error('Use an exact decimal or integer fraction, without an exponent.');
  const negative=s.startsWith('-'),clean=s.replace(/^[+-]/,''),[a,b='']=clean.split('.');return F.add([BigInt((a||'0')+b)*(negative?-1n:1n),10n**BigInt(b.length)],[0n,1n]);
 }
 const label=c=>c.kind==='pixel-domain'?'pixel domain':`${c.kind} ${c.index}`;
 function paintDigit(w,t){const p=w.problem,ctx=$('digit').getContext('2d'),side=p.pixels.length===64?8:p.pixels.length;ctx.fillStyle='#070c10';ctx.fillRect(0,0,256,256);p.pixels.forEach((v,j)=>{const q=j===p.pixel?v+approximate(t):v;const gray=Math.round(q*255);ctx.fillStyle=`rgb(${gray},${gray},${gray})`;ctx.fillRect((j%side)*256/side,Math.floor(j/side)*256/side,256/side,256/side);});ctx.strokeStyle='#9ccedd';ctx.lineWidth=4;ctx.strokeRect((p.pixel%side)*256/side+2,Math.floor(p.pixel/side)*256/side+2,256/side-4,256/side-4);}
 function query(t){
  if(!applied)throw Error('Check a face first.');const w=applied.witness,lo=F.read(w.interval.lower),hi=F.read(w.interval.upper);
  if(F.cmp(t,lo)<0||F.cmp(t,hi)>0)throw Error('Outside this candidate interval. The previous certified delta is retained; no clipping or neighboring-face solve.');
  lastQuery=t;$('query-error').textContent='';$('delta').value=`${t[0]}/${t[1]}`;
  const c=w.coefficients.map(v=>F.add(F.read(v.intercept),F.mul(F.read(v.slope),t))),active=c.flatMap((v,k)=>v[0]>0n?[k]:[]);
  $('query-status').textContent=`Certified delta ≈ ${text(t)}. Pixel ≈ ${text(F.add(PartsCertificate.binaryRatio(w.problem.pixels[w.problem.pixel]),t))}. All exact KKT signs pass.`;$('query-status').dataset.kind='valid';
  $('query-summary').textContent=`${active.length} strictly positive coefficients at this delta. ½ squared-error global optimum for this fixed basis, not for the jointly fitted NMF problem.`;
  const canvas=$('coefficients');canvas.width=Math.max(240,Math.min(520,Math.floor(canvas.parentElement.clientWidth-42)));const cw=canvas.width,ctx=canvas.getContext('2d'),vals=c.map(approximate),max=Math.max(1,...vals);ctx.fillStyle='#172b38';ctx.fillRect(0,0,cw,320);const width=(cw-40)/c.length;
  ctx.fillStyle='#9ccedd';vals.forEach((v,k)=>ctx.fillRect(20+k*width,280-240*v/max,Math.max(2,width-4),240*v/max));ctx.fillStyle='#f3f1e6';ctx.font='14px monospace';ctx.fillText(`scale: 0 to ${max.toPrecision(4)}`,20,20);vals.forEach((v,k)=>{if(k%(cw<400?(c.length>16?4:2):(c.length>16?2:1))===0)ctx.fillText(String(k),20+k*width,298);});paintDigit(w,t);
  window.PARTS_FACE_STATE={...applied,delta:F.serial(t),coefficients:c.map(F.serial)};
 }
 function render(message){
  applied=message;const w=message.witness,lo=F.read(w.interval.lower),hi=F.read(w.interval.upper),isControl=message.control!==undefined;
  $('result').hidden=false;$('context').textContent=isControl?`Applied control: ${controls[message.control]}. Synthetic problem, not a stored UCI row.`:`Applied: rank ${message.rank}, run ${message.run}, test row ${message.row}, pixel ${message.pixel}. Stored numeric coefficients only nominate the face; displayed coefficients are the exact re-solved candidate.`;
  $('bounds').textContent=`${text(lo)} ≤ pixel delta ≤ ${text(hi)}${w.interval.singleton?' · singleton interval':''}`;
  for(const side of ['lower','upper'])$(''+side+'-limits').textContent=w.interval[side+'_limits'].map(label).join(' · ');
  $('face-summary').textContent=`Nominated indices: ${w.problem.support.length?w.problem.support.join(', '):'empty'}. Full source values and exact affine coefficients are preserved in the witness.`;
  $('forms').replaceChildren();for(const v of w.inequalities){const d=document.createElement('div'),h=document.createElement('strong'),p=document.createElement('p');h.textContent=label(v);p.textContent=`${text(F.read(v.intercept))} + (${text(F.read(v.slope))}) × t ≥ 0`;d.append(h,p);$('forms').append(d);}
  $('witness').textContent=JSON.stringify(w,null,2);$('export-status').textContent='';query([0n,1n]);
 }
 function apply(){
  const source=$('problem').value==='source',rank=Number($('rank').value),run=Number($('run').value),row=Number($('row').value),pixel=Number($('pixel').value);
  if(source&&(!$('pixel').value.trim()||!Number.isInteger(pixel)||pixel<0||pixel>63)){ $('error').textContent='Choose an integer pixel index from 0 to 63. The last applied witness is retained.';return;}
  const id=++sequence;if(worker)worker.terminate();worker=new Worker('face-worker.js');$('error').textContent='';$('status').textContent='Checking exact starting KKT and all limiting inequalities in a worker. The last applied context remains visible until this check finishes.';
  worker.onmessage=({data})=>{if(data.id!==sequence)return;if(!data.ok){$('error').textContent=`Exact face refused: ${data.error}. The last applied witness is retained.`;$('status').textContent='Refused without repair.';return;}render(data);$('status').textContent='Exact face verified. This is the applied context; changing selectors alone does not change it.';};
  worker.onerror=()=>{$('error').textContent='The exact worker could not finish. The last applied witness is retained.';$('status').textContent='Worker error.';};
  worker.postMessage(source?{id,rank,run,row,pixel}:{id,control:Number($('problem').value)});
 }
 $('apply').onclick=apply;$('query').onclick=()=>{try{query(parse($('delta').value));}catch(e){$('query-error').textContent=e.message;}};$('delta').onkeydown=e=>{if(e.key==='Enter')$('query').click();};
 for(const side of ['lower','upper'])$(side).onclick=()=>query(F.read(applied.witness.interval[side]));$('zero').onclick=()=>query([0n,1n]);
 $('problem').onchange=()=>{const off=$('problem').value!=='source';for(const id of ['rank','run','row','pixel'])$(id).disabled=off;};
 $('copy').onclick=async()=>{try{await navigator.clipboard.writeText(JSON.stringify(applied.witness));$('export-status').textContent='Witness JSON copied.';}catch(e){$('export-status').textContent='Copy blocked. Download the JSON or select it in the exact witness panel.';}};
 $('download').onclick=()=>{const blob=new Blob([JSON.stringify(applied.witness,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='partslab-face-witness.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('export-status').textContent='Witness JSON download requested.';};
 window.addEventListener('resize',()=>{if(applied&&lastQuery)query(lastQuery);});window.PartsFaceUI={parse,query,apply,approximate};apply();
})();

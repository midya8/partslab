'use strict';
(function(root){
 const Scope='exact supplied binary64; fixed-basis NNLS candidate face; pixel delta';
 const binaryRatio=(typeof module!=='undefined'?require('./certificate.js'):root.PartsCertificate).binaryRatio;
 const zero=[0n,1n],one=[1n,1n];
 function gcd(a,b){if(a<0n)a=-a;while(b){[a,b]=[b,a%b];}return a;}
 function f(n,d=1n){if(d===0n)throw Error('zero denominator');if(d<0n){n=-n;d=-d;}const g=gcd(n,d);return [n/g,d/g];}
 function add(a,b){return f(a[0]*b[1]+b[0]*a[1],a[1]*b[1]);}
 function neg(a){return [-a[0],a[1]];}
 function sub(a,b){return add(a,neg(b));}
 function mul(a,b){return f(a[0]*b[0],a[1]*b[1]);}
 function div(a,b){return f(a[0]*b[1],a[1]*b[0]);}
 function cmp(a,b){const d=a[0]*b[1]-b[0]*a[1];return d<0n?-1:d>0n?1:0;}
 function dot(a,b){return a.reduce((r,v,j)=>add(r,mul(v,b[j])),zero);}
 function serial(a){return {numerator:String(a[0]),denominator:String(a[1])};}
 function read(a){for(const s of [a.numerator,a.denominator])if(typeof s!=='string'||s.length>10000||!/^(0|-?[1-9][0-9]*)$/.test(s))throw Error('canonical bounded integer string required');const n=BigInt(a.numerator),d=BigInt(a.denominator);if(d<=0n)throw Error('positive denominator required');const v=f(n,d);if(v[0]!==n||v[1]!==d)throw Error('reduced rational required');return v;}
 function input(x,b,s,pixel){
  if(!Array.isArray(x)||x.length<1||x.length>256||!Array.isArray(b)||b.length<1||b.length>64||b.some(r=>!Array.isArray(r)||r.length!==x.length))throw Error('inconsistent dimensions');
  if(x.some(v=>typeof v!=='number'||!Number.isFinite(v)||v<0||v>1))throw Error('pixels must be in [0,1]');
  if(!Number.isInteger(pixel)||pixel<0||pixel>=x.length)throw Error('integer pixel index required');
  if(!Array.isArray(s)||s.some((k,i)=>!Number.isInteger(k)||k<0||k>=b.length||(i>0&&k<=s[i-1])))throw Error('sorted distinct integer support required');
  return {x:x.map(binaryRatio),b:b.map(r=>r.map(binaryRatio))};
 }
 function solve(g,rhs){
  // Fraction-free Bareiss elimination. Every input is dyadic, so the maximum
  // denominator is a common multiple. No rounding, regularization or pivot fix.
  const n=g.length;if(n===0)return [];
  const raw=g.map((r,i)=>[...r,...rhs[i]]),scale=raw.flat().reduce((m,v)=>v[1]>m?v[1]:m,1n);
  const a=raw.map(r=>r.map(v=>{if(scale%v[1]!==0n)throw Error('nondyadic solve');return v[0]*(scale/v[1]);}));
  let previous=1n;
  for(let k=0;k<n;k++){
   const pivot=a.findIndex((r,i)=>i>=k&&r[k]!==0n);if(pivot<0)throw Error('singular selected Gram block');
   [a[k],a[pivot]]=[a[pivot],a[k]];const d=a[k][k];
   for(let i=k+1;i<n;i++){
    for(let j=k+1;j<n+2;j++){const value=d*a[i][j]-a[i][k]*a[k][j];if(value%previous!==0n)throw Error('inexact Bareiss division');a[i][j]=value/previous;}
    a[i][k]=0n;
   }
   previous=d;
  }
  const ans=Array.from({length:n},()=>[zero,zero]);
  for(let i=n-1;i>=0;i--)for(let q=0;q<2;q++){
   let r=f(a[i][n+q]);for(let j=i+1;j<n;j++)r=sub(r,mul(f(a[i][j]),ans[j][q]));ans[i][q]=div(r,f(a[i][i]));
  }
  return ans;
 }
 function forms(x,b,s,pixel,c0,c1){
  const scale=[...x,...b.flat()].reduce((m,v)=>v[1]>m?v[1]:m,1n),xi=x.map(v=>v[0]*(scale/v[1])),bi=b.map(row=>row.map(v=>v[0]*(scale/v[1])));
  const common=[...c0,...c1].reduce((m,v)=>m/gcd(m,v[1])*v[1],1n),ci=c0.map(v=>v[0]*(common/v[1])),di=c1.map(v=>v[0]*(common/v[1]));
  const r0=xi.map((v,j)=>bi.reduce((r,row,k)=>r+ci[k]*row[j],0n)-v*common);
  const r1=xi.map((_,j)=>bi.reduce((r,row,k)=>r+di[k]*row[j],0n)-(j===pixel?common*scale:0n));
  const denominator=common*scale*scale,grad=bi.map(row=>[f(row.reduce((r,v,j)=>r+v*r0[j],0n),denominator),f(row.reduce((r,v,j)=>r+v*r1[j],0n),denominator)]),selected=new Set(s);
  if(s.some(k=>grad[k].some(v=>v[0]!==0n)))throw Error('selected stationarity failure');
  if(b.some((_,k)=>!selected.has(k)&&(c0[k][0]!==0n||c1[k][0]!==0n)))throw Error('unselected coefficient failure');
  return [...s.map(k=>['coefficient',k,c0[k],c1[k]]),...b.flatMap((_,k)=>selected.has(k)?[]:[['gradient',k,...grad[k]]])];
 }
 function interval(fs,x,pixel){
  const domainLo=neg(x[pixel]),domainHi=sub(one,x[pixel]);let lo=domainLo,hi=domainHi;
  for(const [kind,k,a,b] of fs){if(a[0]<0n)throw Error('candidate fails exact KKT at starting point');if(b[0]!==0n){const t=div(neg(a),b);if(b[0]>0n&&cmp(t,lo)>0)lo=t;if(b[0]<0n&&cmp(t,hi)<0)hi=t;}}
  if(lo[0]>0n||hi[0]<0n)throw Error('starting point excluded');
  const limits=(t,domain)=>{const arr=fs.filter(([kind,k,a,b])=>b[0]!==0n&&add(a,mul(b,t))[0]===0n).map(([kind,index])=>({kind,index}));if(cmp(t,domain)===0)arr.push({kind:'pixel-domain',index:pixel});return arr;};
  return {lower:serial(lo),upper:serial(hi),lower_limits:limits(lo,domainLo),upper_limits:limits(hi,domainHi),singleton:cmp(lo,hi)===0};
 }
 const serializeForms=fs=>fs.map(([kind,index,a,b])=>({kind,index,intercept:serial(a),slope:serial(b)}));
 function response(pixels,basis,support,pixel){
  const {x,b}=input(pixels,basis,support,pixel),scale=[...x,...b.flat()].reduce((m,v)=>v[1]>m?v[1]:m,1n),xi=x.map(v=>v[0]*(scale/v[1])),bi=b.map(row=>row.map(v=>v[0]*(scale/v[1])));
  const idot=(a,z)=>a.reduce((v,k,j)=>v+k*z[j],0n),g=support.map(k=>support.map(l=>f(idot(bi[k],bi[l]),scale*scale))),rhs=support.map(k=>[f(idot(bi[k],xi),scale*scale),f(bi[k][pixel],scale)]),ans=solve(g,rhs),c0=b.map(()=>zero),c1=b.map(()=>zero);
  support.forEach((k,i)=>{[c0[k],c1[k]]=ans[i];});const fs=forms(x,b,support,pixel,c0,c1);
  return {schema:1,scope:Scope,problem:{pixels:pixels.slice(),basis:basis.map(r=>r.slice()),support:support.slice(),pixel},coefficients:c0.map((a,k)=>({intercept:serial(a),slope:serial(c1[k])})),inequalities:serializeForms(fs),interval:interval(fs,x,pixel)};
 }
 function equal(a,b){if(typeof a!==typeof b)return false;if(a===null||b===null)return a===b;if(typeof a!=='object')return a===b;if(Array.isArray(a)!==Array.isArray(b))return false;const k=Object.keys(a);return k.length===Object.keys(b).length&&k.every(n=>Object.hasOwn(b,n)&&equal(a[n],b[n]));}
 function verify(w){try{
  if(w.schema!==1||w.scope!==Scope)return false;const p=w.problem,{x,b}=input(p.pixels,p.basis,p.support,p.pixel);
  if(!Array.isArray(w.coefficients)||w.coefficients.length!==b.length)return false;
  const c0=w.coefficients.map(c=>read(c.intercept)),c1=w.coefficients.map(c=>read(c.slope));
  if(!equal(w.coefficients,c0.map((a,k)=>({intercept:serial(a),slope:serial(c1[k])}))))return false;
  const fs=forms(x,b,p.support,p.pixel,c0,c1);return equal(w.inequalities,serializeForms(fs))&&equal(w.interval,interval(fs,x,p.pixel));
 }catch(e){return false;}}
 const api={response,verify,read,add,mul,cmp,serial};if(typeof module!=='undefined')module.exports=api;else root.PartsFace=api;
})(globalThis);

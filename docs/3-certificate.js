'use strict';
(function(root){
 const SCOPE='fixed-basis NNLS; exact supplied binary64 values',dv=new DataView(new ArrayBuffer(8));
 function binaryRatio(v){
  if(typeof v!=='number'||!Number.isFinite(v)||v<0)throw Error('finite nonnegative binary64 values required');
  if(v===0)return [0n,1n];
  dv.setFloat64(0,v,false);const bits=dv.getBigUint64(0,false),biased=Number((bits>>52n)&2047n);
  let m=bits&((1n<<52n)-1n),exp=biased===0?-1074:biased-1075;if(biased!==0)m|=1n<<52n;
  while(exp<0&&(m&1n)===0n){m>>=1n;exp++;}
  return exp>=0?[m<<BigInt(exp),1n]:[m,1n<<BigInt(-exp)];
 }
 function problem(x,b,c){
  if(!Array.isArray(x)||!Array.isArray(b)||!Array.isArray(c)||x.length<1||x.length>256||b.length<1||b.length>64||c.length!==b.length||b.some(r=>!Array.isArray(r)||r.length!==x.length))throw Error('inconsistent certificate dimensions');
  const all=[...x,...c,...b.flat()].map(binaryRatio),s=all.reduce((m,r)=>r[1]>m?r[1]:m,1n),integer=v=>{const [n,d]=binaryRatio(v);return n*(s/d);};
  return {x:x.map(integer),b:b.map(r=>r.map(integer)),c:c.map(integer),s};
 }
 function gcd(a,b){if(a<0n)a=-a;while(b){const t=a%b;a=b;b=t;}return a;}
 function frac(n,d){if(d<=0n)throw Error('positive denominator required');const g=gcd(n,d);return {numerator:String(n/g),denominator:String(d/g)};}
 function integer(s){if(typeof s!=='string'||s.length>10000||!/^(0|-?[1-9][0-9]*)$/.test(s))throw Error('canonical bounded integer string required');return BigInt(s);}
 function rf(f){const n=integer(f.numerator),d=integer(f.denominator);if(d<=0n)throw Error('positive denominator required');return [n,d];}
 function decimalBound(n,d,upper=false,digits=6){
  if(n<0n||d<=0n||!Number.isInteger(digits)||digits<1||digits>20)throw Error('invalid decimal bound');if(n===0n)return '0';
  let exp=String(n).length-String(d).length;
  if(exp>=0?n<d*10n**BigInt(exp):n*10n**BigInt(-exp)<d)exp--;
  const power=digits-1-exp,a=power>=0?n*10n**BigInt(power):n,b=power>=0?d:d*10n**BigInt(-power);let q=a/b;
  if(upper&&a%b)q++;if(q>=10n**BigInt(digits)){q/=10n;exp++;}const text=String(q).padStart(digits,'0');return text[0]+(digits>1?'.'+text.slice(1):'')+'e'+exp;
 }
 function display(p,d,g){const [pn,pd]=rf(p),[dn,dd]=rf(d),[gn,gd]=rf(g);return {optimal_objective_lower:decimalBound(dn,dd),optimal_objective_upper:decimalBound(pn,pd,true),suboptimality_upper:decimalBound(gn,gd,true)};}
 function certify(pixels,basis,coefficients){
  const {x,b,c,s}=problem(pixels,basis,coefficients),R=s*s,xr=x.map(v=>v*s),r=x.map((_,j)=>xr[j]-b.reduce((sum,row,k)=>sum+c[k]*row[j],0n));let N=0n,D=1n;
  for(const row of b){const sum=row.reduce((a,v)=>a+v,0n),cross=row.reduce((a,v,j)=>a+v*r[j],0n);if(sum&&cross*D>N*sum){N=cross;D=sum;}}
  let y=r.map(v=>v*D-N),yd=R*D,dn=2n*yd*y.reduce((a,v,j)=>a+xr[j]*v,0n)-R*y.reduce((a,v)=>a+v*v,0n),dd=2n*R*yd*yd,source='shifted-residual';
  if(dn<0n){y=x.map(()=>0n);yd=1n;dn=0n;dd=1n;source='zero-fallback';}
  const pn=r.reduce((a,v)=>a+v*v,0n),pd=2n*R*R,gn=pn*dd-dn*pd,gd=pd*dd;if(gn<0n)throw Error('weak duality violated');
  const p=frac(pn,pd),d=frac(dn,dd),g=frac(gn,gd);
  return {schema:1,scope:SCOPE,problem:{pixels:pixels.slice(),basis:basis.map(r=>r.slice()),coefficients:coefficients.slice()},dual_numerators:y.map(String),dual_denominator:String(yd),dual_source:source,primal:p,dual:d,gap:g,display:display(p,d,g)};
 }
 function verify(w){try{
  if(w.schema!==1||w.scope!==SCOPE)return false;const q=w.problem,{x,b,c,s}=problem(q.pixels,q.basis,q.coefficients),yd=integer(w.dual_denominator),y=w.dual_numerators.map(integer);if(yd<=0n||y.length!==x.length)return false;
  for(const row of b)if(row.reduce((a,v,j)=>a+v*y[j],0n)>0n)return false;
  const R=s*s,r=x.map((v,j)=>v*s-b.reduce((a,row,k)=>a+c[k]*row[j],0n)),pn=r.reduce((a,v)=>a+v*v,0n),pd=2n*R*R;
  const dn=2n*yd*y.reduce((a,v,j)=>a+x[j]*v,0n)-s*y.reduce((a,v)=>a+v*v,0n),dd=2n*s*yd*yd;
  const [wpn,wpd]=rf(w.primal),[wdn,wdd]=rf(w.dual),[wgn,wgd]=rf(w.gap);
  if(dn<0n||wgn<0n||wpn*pd!==pn*wpd||wdn*dd!==dn*wdd||wgn*wpd*wdd!==(wpn*wdd-wdn*wpd)*wgd)return false;
  const v=display(w.primal,w.dual,w.gap);return Object.keys(v).every(k=>w.display[k]===v[k])&&Object.keys(w.display).length===3;
 }catch(e){return false;}}
 const api={certify,verify,binaryRatio,decimalBound};if(typeof module!=='undefined')module.exports=api;else root.PartsCertificate=api;
})(globalThis);

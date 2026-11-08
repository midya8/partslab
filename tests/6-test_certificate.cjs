'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const C=require('../docs/certificate.js'),artifact=require('../partslab/artifacts/certificates.json'),ctx={};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../docs/data.js'),'utf8'),ctx);const D=ctx.PARTS_DATA;
test('all 120 native exact rational witnesses and five controls agree with independent BigInt implementation',()=>{
 for(const v of artifact.cases){const a=D.arms[String(v.rank)],basis=Array.from(a.runs[v.run].basis,r=>Array.from(r)),pixels=Array.from(D.examples[v.row].pixels),coefficients=Array.from(D.reference[String(v.rank)][v.run][v.row].coefficients),w=C.certify(pixels,basis,coefficients);assert.equal(C.verify(w),true);delete w.problem;assert.deepEqual(w,v.certificate);}
 for(const w of artifact.controls){assert.equal(C.verify(w),true);const p=w.problem;assert.deepEqual(C.certify(p.pixels,p.basis,p.coefficients),w);}
});
test('binary64 conversion including subnormals, powers, ordinary decimals and nonnegative guards',()=>{
 for(const [v,n,d] of [[0,0n,1n],[.5,1n,2n],[.1,3602879701896397n,36028797018963968n],[Number.MIN_VALUE,1n,1n<<1074n],[2**500,1n<<500n,1n]])assert.deepEqual(C.binaryRatio(v),[n,d]);
 for(const v of [NaN,Infinity,-1,'1'])assert.throws(()=>C.binaryRatio(v));
 const sub=C.certify([Number.MIN_VALUE],[[1]],[0]);assert.equal(C.verify(sub),true);
});
test('feasibility, fractions, display, scope and problem tampering are rejected',()=>{
 const original=C.certify([.8,.3],[[.2,.7]],[.4]);assert.equal(C.verify(original),true);
 for(const k of ['primal','dual','gap']){const w=structuredClone(original);w[k].numerator=String(BigInt(w[k].numerator)+1n);assert.equal(C.verify(w),false);}
 for(const [k,v] of [['dual_denominator','0'],['dual_numerators',['1','1']],['scope','joint global NMF']]){const w=structuredClone(original);w[k]=v;assert.equal(C.verify(w),false);}
 const w=structuredClone(original);w.display.suboptimality_upper='0';assert.equal(C.verify(w),false);assert.equal(C.verify({}),false);assert.equal(C.verify(null),false);
});
test('outward decimal formatting and duplicate zero-gap control do not imply unique coefficients',()=>{
 assert.equal(C.decimalBound(1n,3n,false),'3.33333e-1');assert.equal(C.decimalBound(1n,3n,true),'3.33334e-1');assert.equal(C.decimalBound(9999999n,1000000n,true),'1.00000e1');
 const a=C.certify([1,0],[[1,0],[1,0]],[1,0]),b=C.certify([1,0],[[1,0],[1,0]],[0,1]);assert.equal(a.gap.numerator,'0');assert.equal(b.gap.numerator,'0');assert.notDeepEqual(a.problem.coefficients,b.problem.coefficients);
});

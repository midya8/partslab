'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const face=require('../docs/face.js'),native=JSON.parse(fs.readFileSync('partslab/artifacts/face-reference.json','utf8'));
test('all54 complete native Fraction responses match scalar BigInt including limits',()=>{
 assert.equal(native.cases.length,54);
 for(const c of native.cases){const w=c.witness,p=w.problem,actual=face.response(p.pixels,p.basis,p.support,p.pixel);assert.deepEqual(actual,w);assert(face.verify(w));}
 for(const w of native.controls){const p=w.problem;assert.deepEqual(face.response(p.pixels,p.basis,p.support,p.pixel),w);assert(face.verify(w));}
});
test('BigInt exact face refusals and independent witness tampering',()=>{
 assert.throws(()=>face.response([.5],[[1],[1]],[0,1],0),/singular/);
 assert.throws(()=>face.response([.5],[[1]],[],0),/starting/);
 for(const s of [[true],[0,0],[1,0],[-1],[2]])assert.throws(()=>face.response([.5],[[1],[1]],s,0));
 const w=face.response([.5,.5],[[1,1],[1,0]],[0],0);
 for(const change of ['bound','limit','slope','canonical','schema']){
  const v=structuredClone(w);
  if(change==='bound')v.interval.upper.numerator='1';
  if(change==='limit')v.interval.upper_limits=[];
  if(change==='slope')v.coefficients[0].slope.numerator='0';
  if(change==='canonical')v.coefficients[0].slope={numerator:'2',denominator:'4'};
  if(change==='schema')v.schema=true;
  assert(!face.verify(v),change);
 }
});

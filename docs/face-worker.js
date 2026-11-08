'use strict';
importScripts('certificate.js','face.js','data.js');
onmessage=event=>{
 const {id,rank,run,row,pixel,control}=event.data;
 try{
  if(control!==undefined){
   const args=[[[.5,.5],[[1,1],[1,0]],[0],0],[[0,0],[[1,0]],[],0],[[.5],[[1],[1]],[0],0],[[.5,.5],[[1,0],[0,1]],[0,1],0]][control];
   if(!args)throw Error('unknown control');const witness=PartsFace.response(...args);if(!PartsFace.verify(witness))throw Error('control verification failed');postMessage({id,ok:true,control,witness});return;
  }
  if(!['8','16','32'].includes(String(rank))||![0,1].includes(run)||!Number.isInteger(row)||row<0||row>=20)throw Error('unknown stored source context');
  const data=PARTS_DATA,basis=data.arms[String(rank)].runs[run].basis,pixels=data.examples[row].pixels;
  const support=data.reference[String(rank)][run][row].coefficients.flatMap((v,k)=>v>0?[k]:[]);
  const witness=PartsFace.response(pixels,basis,support,pixel);
  if(!PartsFace.verify(witness))throw Error('internal exact verification failed');
  postMessage({id,ok:true,rank:Number(rank),run,row,pixel,witness});
 }catch(e){postMessage({id,ok:false,error:e.message});}
};

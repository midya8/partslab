'use strict';
(function(root){
 function predict(pixels,arm,run,mean,maxSweeps=1000,tol=1e-8){
  if(!Array.isArray(pixels)||pixels.length!==64||pixels.some(x=>typeof x!=='number'||!Number.isFinite(x)||x<0||x>1))throw Error('64 finite pixels in [0,1] required');
  if(!Number.isInteger(maxSweeps)||maxSweeps<0||!Number.isFinite(tol)||tol<0)throw Error('invalid inference budget');
  const b=run.basis,k=b.length,a=Array(k).fill(0),cross=b.map(row=>row.reduce((s,v,j)=>s+v*pixels[j],0)),gram=b.map(r=>b.map(t=>r.reduce((s,v,j)=>s+v*t[j],0))),threshold=tol*Math.max(1,...cross.map(Math.abs));
  let reconstruction,sweeps=0,pg=Infinity,converged=false;
  for(let it=0;it<=maxSweeps;it++){
   reconstruction=pixels.map((_,j)=>a.reduce((s,v,i)=>s+v*b[i][j],0));
   const g=b.map(row=>row.reduce((s,v,j)=>s+(reconstruction[j]-pixels[j])*v,0));pg=Math.max(...g.map((v,j)=>(a[j]>0||v<0)?Math.abs(v):0));converged=pg<=threshold;
   if(converged||it===maxSweeps)break;
   const grad=gram.map((r,j)=>r.reduce((s,v,i)=>s+a[i]*v,0)-cross[j]);
   for(let j=0;j<k;j++){const old=a[j];a[j]=gram[j][j]>0?Math.max(0,old-grad[j]/gram[j][j]):0;const delta=a[j]-old;for(let z=0;z<k;z++)grad[z]+=delta*gram[j][z];}sweeps++;
  }
  const project=(components,center)=>{const y=pixels.map((v,j)=>v-center[j]),weights=components.map(r=>r.reduce((s,v,j)=>s+v*y[j],0));return pixels.map((_,j)=>center[j]+weights.reduce((s,v,i)=>s+v*components[i][j],0));};
  const mse=r=>r.reduce((s,v,j)=>s+(v-pixels[j])**2,0)/64,pca=project(arm.pca_components,mean),svd=project(arm.svd_components,Array(64).fill(0));
  return {rank:k,seed:run.seed,coefficients:a,reconstruction,nnls_converged:converged,nnls_sweeps:sweeps,projected_gradient_inf:pg,threshold,nmf_mse:mse(reconstruction),pca_reconstruction:pca,pca_mse:mse(pca),svd_reconstruction:svd,svd_mse:mse(svd),mean_mse:mse(mean)};
 }
 const api={predict};if(typeof module!=='undefined')module.exports=api;else root.PartsInference=api;
})(globalThis);

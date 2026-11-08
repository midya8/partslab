import argparse
from .io import ROOT,read,write,dumps
from .experiment import train,evaluate,predict,PROTOCOL
from .audit import controls

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    t=s.add_parser('train');t.add_argument('--out',required=True);t.add_argument('--guard',required=True)
    e=s.add_parser('evaluate');e.add_argument('--models',default=str(ROOT/'artifacts/models.json'));e.add_argument('--guard',default=str(ROOT/'artifacts/guard.json'));e.add_argument('--out',required=True)
    q=s.add_parser('predict');q.add_argument('--pixels',required=True);q.add_argument('--rank',type=int,choices=PROTOCOL['ranks'],default=16);q.add_argument('--out',required=True);q.add_argument('--models',default=str(ROOT/'artifacts/models.json'))
    a=s.add_parser('audit');a.add_argument('--out',required=True)
    s.add_parser('demo');args=p.parse_args()
    if args.command=='train':m,g=train();write(args.out,m);write(args.guard,g)
    elif args.command=='evaluate':write(args.out,evaluate(read(args.models),read(args.guard)))
    elif args.command=='predict':write(args.out,predict(read(args.pixels),read(args.models),args.rank))
    elif args.command=='audit':write(args.out,controls())
    else:
        r=read(ROOT/'artifacts/report.json');print(dumps({'test_rows':r['test_rows'],'mean_mse':r['mean_mse'],'arms':{k:{'chosen_seed':v['chosen_seed'],'chosen_nmf_mse':next(x['mse'] for x in v['nmf_runs'] if x['seed']==v['chosen_seed']),'pca_mse':v['pca_mse'],'svd_mse':v['svd_mse']} for k,v in r['arms'].items()}}))
if __name__=='__main__':main()

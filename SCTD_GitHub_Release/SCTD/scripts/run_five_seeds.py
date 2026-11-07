"""Run independent SCTD training runs with VAL-only selection; store logs and final TEST JSON."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['mosi','mosei'],required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,54,58,101,2334])
    p.add_argument('--n_epochs',type=int,default=50)
    p.add_argument('--bert_path',default='./bert-base-uncased')
    p.add_argument('--train_batch_size',type=int,default=None)
    args=p.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True)
    results=[]
    for seed in args.seeds:
        cmd=[sys.executable,str(ROOT/'main.py'),'--dataset',args.dataset,'--gpu',str(args.gpu),'--seed',str(seed),
             '--n_epochs',str(args.n_epochs),'--bert_path',args.bert_path,'--no_epoch_test','--select_metric','mse']
        if args.train_batch_size:cmd+=['--train_batch_size',str(args.train_batch_size)]
        elif args.dataset=='mosei':cmd+=['--train_batch_size','128']
        print('Running:',' '.join(cmd),flush=True)
        run=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
        path=out/f'{args.dataset}_seed{seed}.log';path.write_text(run.stdout,encoding='utf-8')
        if run.returncode:
            raise SystemExit(f'Seed {seed} failed with exit code {run.returncode}. See {path}')
        lines=[s for s in run.stdout.splitlines() if s.startswith('Final TEST: ')]
        if not lines:
            raise RuntimeError(f'Seed {seed} did not output Final TEST. See {path}')
        scores=json.loads(lines[-1].split('Final TEST: ',1)[1]);scores['seed']=seed;results.append(scores)
        print('Final TEST',seed,scores,flush=True)
    (out/f'{args.dataset}_five_seeds.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    for key in ('Acc_2','F1','Acc_7','Corr','MAE'):
        vals=[r[key] for r in results if r.get(key) is not None]
        if not vals:continue
        import statistics
        mean=statistics.mean(vals);std=statistics.stdev(vals) if len(vals)>1 else 0.
        mul=100 if key in ('Acc_2','F1','Acc_7') else 1
        print(f'{key}: {mul*mean:.4f} ± {mul*std:.4f} (sample SD, {len(vals)} seeds)')

if __name__=='__main__':main()

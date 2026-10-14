"""Fresh fixed-pool paired gate against both neural incumbent and old v2."""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
from collect_bluff_counterfactual import checkpoint_spec
from select_bluff_residual_cpu import evaluate
from train_bluff_league import paired_interval,summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--incumbent',type=Path,default=Path('bluffSDK/models/policy.json'))
    parser.add_argument('--v2',type=Path,default=Path('bluffSDK/models/policy-v2.json'))
    parser.add_argument('--blocks',type=int,default=512)
    parser.add_argument('--seed-start',type=int,required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--opponent-pool',type=Path,help='reuse a frozen selection pool JSON')
    args=parser.parse_args();os.environ['BLUFF_FAST_OBSERVATIONS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    docs=[json.loads(path.read_text()) for path in (args.candidate,args.incumbent,args.v2)];specs=[checkpoint_spec(doc) for doc in docs]
    reference=[{'kind':kind} for kind in ('starter','competitive','heuristic','random','truthful','aggressive')]+specs[1:]
    reference += [checkpoint_spec(json.loads(Path('bluffSDK/models/policy-v1.json').read_text()))]
    reference += [{'kind':'bluffer','bluff_rate':bluff,'call_rate':call} for bluff,call in ((.15,.03),(.35,.1),(.5,.35))]
    if args.opponent_pool:
        frozen=json.loads(args.opponent_pool.read_text())
        reference=frozen['opponent_pool'] if isinstance(frozen,dict) else frozen
        if not isinstance(reference,list) or not reference:
            raise ValueError('frozen opponent pool must be a nonempty list')
    regimes=[(mode,reference,mode) for mode in ('league','competitive','starter','truthful','bluffer','aggressive')]
    regimes += [('neural_selfplay',[specs[1]],'league'),('v2_selfplay',[specs[2]],'league')]
    regimes += [(label,[{'kind':'bluffer','bluff_rate':bluff,'call_rate':.35}],'league') for label,bluff in (('weakspot_bluff01_call035',.1),('weakspot_bluff065_call035',.65))]
    seeds=list(range(args.seed_start,args.seed_start+args.blocks));report={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for label,opponents,mode in regimes:
            rows=evaluate(pool,specs,seeds,opponents,mode)
            report[label]={'candidate':summarize(rows[0]),'incumbent_neural':summarize(rows[1]),'retained_v2':summarize(rows[2]),'paired_gain_vs_neural':paired_interval(rows[0],rows[1]),'paired_gain_vs_v2':paired_interval(rows[0],rows[2])}
            print(json.dumps({'mode':label,**report[label]}),flush=True)
    args.output.write_text(json.dumps({'seed_start':args.seed_start,'opponent_pool':reference,'candidate_training':docs[0]['training'],'evaluation':report},indent=2)+'\n')


if __name__=='__main__':main()

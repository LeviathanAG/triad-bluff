"""Fresh fixed-pool paired gate against both neural incumbent and old v2."""
import argparse
import concurrent.futures
import json
import hashlib
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
    parser.add_argument('--regime',action='append',default=[],help='run only named predefined regimes; repeat to select more than one')
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
    if args.regime:
        unknown=set(args.regime)-{row[0] for row in regimes}
        if unknown:parser.error('unknown regimes: '+','.join(sorted(unknown)))
        regimes=[row for row in regimes if row[0] in args.regime]
    seeds=list(range(args.seed_start,args.seed_start+args.blocks));report={};block_rows={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for label,opponents,mode in regimes:
            rows=evaluate(pool,specs,seeds,opponents,mode)
            block_rows[label]={'candidate':rows[0],'incumbent_neural':rows[1],'retained_v2':rows[2]}
            report[label]={'candidate':summarize(rows[0]),'incumbent_neural':summarize(rows[1]),'retained_v2':summarize(rows[2]),'paired_gain_vs_neural':paired_interval(rows[0],rows[1]),'paired_gain_vs_v2':paired_interval(rows[0],rows[2])}
            print(json.dumps({'mode':label,**report[label]}),flush=True)
    args.output.write_text(json.dumps({'seed_start':args.seed_start,'opponent_pool':reference,'candidate_training':docs[0]['training'],'evaluation':report},indent=2)+'\n')
    sidecar={'sdk_game_commit':'96da8042c285af4e4f43655f2ad99592bd49b8e6',
             'game_config':{'players':4,'decks':1,'max_turns':1000,'position_points':[5,3,2,1]},
             'seed_ids':seeds,'sampling_unit':'one seed, all four seats averaged; ordered seed_ids match each block array',
             'checkpoint_order':['candidate','incumbent_neural','retained_v2'],
             'checkpoint_canonical_sha256':[hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':')).encode()).hexdigest() for document in docs],
             'opponent_pool_canonical_sha256':hashlib.sha256(json.dumps(reference,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
             'rows':block_rows}
    args.output.with_name(args.output.stem+'-blocks.json').write_text(json.dumps(sidecar,indent=2)+'\n')


if __name__=='__main__':main()

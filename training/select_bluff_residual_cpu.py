"""CPU actual-game selection for GPU-exported neural candidates."""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
from train_bluff_league import evaluate_job, paired_interval, summarize
from evaluate_bluff_checkpoint import combine
from collect_bluff_counterfactual import checkpoint_spec
from bluff_mlp_policy import Policy


def evaluate(pool, specifications, seeds, opponents, mode):
    chunks=[seeds[start:start+16] for start in range(0,len(seeds),16)]
    jobs=[(spec,chunk,opponents,mode) for spec in specifications for chunk in chunks]
    raw=list(pool.map(evaluate_job,jobs));n=len(chunks)
    return [combine(raw[index*n:(index+1)*n]) for index in range(len(specifications))]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--linear-backup',type=Path,required=True)
    parser.add_argument('--previous-v1',type=Path)
    parser.add_argument('--archive',type=Path,action='append',default=[],help='additional fixed learned rivals')
    parser.add_argument('--validation-blocks',type=int,default=128)
    parser.add_argument('--test-blocks',type=int,default=256)
    parser.add_argument('--validation-seed-start',type=int,default=65_000_000)
    parser.add_argument('--test-seed-start',type=int,default=66_000_000)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--robust-selection',action='store_true',help='select using five fixed rival compositions with a per-regime loss guard')
    parser.add_argument('--max-selection-regression',type=float,default=.025)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.validation_blocks,args.test_blocks,args.workers)<1:parser.error('positive counts required')
    os.environ['BLUFF_FAST_OBSERVATIONS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    args.output.mkdir(parents=True,exist_ok=True)
    fit=json.loads((args.candidates/'fit-summary.json').read_text())
    incumbent_document=fit['baseline_checkpoint'];incumbent=checkpoint_spec(incumbent_document)
    linear_document=json.loads(args.linear_backup.read_text());linear=checkpoint_spec(linear_document)
    documents=[json.loads((args.candidates/f'candidate-{index}.json').read_text()) for index in range(fit['candidate_count'])]
    for document in documents:Policy(document)
    reference=[{'kind':kind} for kind in ('starter','competitive','heuristic','random','truthful','aggressive')]+[incumbent,linear]
    if args.previous_v1:reference.append(checkpoint_spec(json.loads(args.previous_v1.read_text())))
    reference += [checkpoint_spec(json.loads(path.read_text())) for path in args.archive]
    reference += [{'kind':'bluffer','bluff_rate':bluff,'call_rate':call} for bluff,call in ((.1,.0),(.1,.35),(.35,.1),(.65,.35),(.95,.0),(.95,.35))]
    specs=[incumbent]+[checkpoint_spec(document) for document in documents]
    seeds=list(range(args.validation_seed_start,args.validation_seed_start+args.validation_blocks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=evaluate(pool,specs,seeds,reference,'league')
        chosen=max(range(len(results)),key=lambda index:results[index]['points'])
        selection={'selected':chosen,'seed_start':args.validation_seed_start,'scores':[summarize(row) for row in results],'paired_gains':[paired_interval(row,results[0]) for row in results[1:]],'opponent_pool':reference}
        if args.robust_selection:
            # These seeds select a fixed checkpoint only. Fresh final seeds are
            # never consulted during candidate or objective selection.
            regime_results={'league':results}
            for label,opponents,mode in [('neural_selfplay',[incumbent],'league'),
                                         ('competitive',reference,'competitive'),
                                         ('truthful',reference,'truthful'),
                                         ('bluffer',reference,'bluffer')]:
                regime_results[label]=evaluate(pool,specs,seeds,opponents,mode)
                print(json.dumps({'selection_regime':label,'scores':[summarize(row) for row in regime_results[label]]}),flush=True)
            gains={label:[summarize(row)['points']-summarize(rows[0])['points'] for row in rows]
                   for label,rows in regime_results.items()}
            utilities=[sum(values[index] for values in gains.values())/len(gains) for index in range(len(specs))]
            feasible=[index for index in range(len(specs))
                      if min(values[index] for values in gains.values())>=-args.max_selection_regression]
            chosen=max(feasible,key=lambda index:utilities[index])
            selection.update(selected=chosen,selection_rule='equal mean gain across five predeclared regimes, subject to per-regime mean regression guard; baseline wins ties',
                             max_selection_regression=args.max_selection_regression,
                             regime_scores={label:[summarize(row) for row in rows] for label,rows in regime_results.items()},
                             regime_paired_gains={label:[paired_interval(row,rows[0]) for row in rows[1:]] for label,rows in regime_results.items()},
                             feasible_candidates=feasible,mean_gain_utilities=utilities)
        (args.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        print(json.dumps({'actual_game_selection':{key:value for key,value in selection.items() if key!='opponent_pool'}}),flush=True)
        selected_document=documents[chosen-1] if chosen else incumbent_document
        (args.output/'bluff-policy-mlp.json').write_text(json.dumps(selected_document,indent=2)+'\n')
        selected=specs[chosen];final_seeds=list(range(args.test_seed_start,args.test_seed_start+args.test_blocks));report={}
        regimes=[('league',reference,'league'),('competitive',reference,'competitive'),('starter',reference,'starter'),('truthful',reference,'truthful'),('bluffer',reference,'bluffer'),('aggressive',reference,'aggressive'),('incumbent_selfplay',[incumbent],'league')]
        regimes += [(label,[{'kind':'bluffer','bluff_rate':bluff,'call_rate':.35}],'league') for label,bluff in (('weakspot_bluff01_call035',.1),('weakspot_bluff065_call035',.65))]
        for label,opponents,mode in regimes:
            rows=evaluate(pool,[selected,incumbent,linear],final_seeds,opponents,mode)
            report[label]={'candidate':summarize(rows[0]),'incumbent_neural':summarize(rows[1]),'retained_v2':summarize(rows[2]),'paired_gain_vs_neural':paired_interval(rows[0],rows[1]),'paired_gain_vs_v2':paired_interval(rows[0],rows[2])}
            print(json.dumps({'final_regime':label,**report[label]}),flush=True)
        (args.output/'evaluation.json').write_text(json.dumps({'seed_start':args.test_seed_start,'candidate_training':selected_document['training'],'baseline_training':incumbent_document['training'],'evaluation':report},indent=2)+'\n')


if __name__=='__main__':main()

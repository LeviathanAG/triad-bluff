"""Benchmark trained Bluff against actual SDK starter and adaptive mixtures."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
from train_bluff_league import evaluate_job, paired_interval, summarize
from bluff_policy import INITIAL_WEIGHTS
from bluff_policy_v2 import INITIAL_WEIGHTS as V2_INITIAL


def combine(parts):
 points=[v for p in parts for v in p['block_points']]
 wins=[v for p in parts for v in p['block_wins']]
 return {'points':sum(points)/len(points),'win_rate':sum(wins)/len(wins),'games':4*len(points),'block_points':points,'block_wins':wins}


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--model',type=Path,required=True)
 parser.add_argument('--blocks',type=int,default=256)
 parser.add_argument('--workers',type=int,default=4)
 parser.add_argument('--output',type=Path,default=Path(__file__).parent/'bluff-v1-verification.json')
 args=parser.parse_args()
 document=json.loads(args.model.read_text())
 candidate={'kind':'v1','weights':document['weights']};starter={'kind':'starter'}
 reference=[{'kind':k} for k in ('starter','heuristic','truthful','aggressive','random')]+[
  {'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v2','weights':V2_INITIAL},candidate]+[
  {'kind':'bluffer','bluff_rate':b,'call_rate':c} for b,c in ((.15,.03),(.35,.1),(.5,.35))]
 modes=('league','starter','truthful','bluffer','aggressive')
 seeds=list(range(13_000_000,13_000_000+args.blocks));chunks=[seeds[i:i+32] for i in range(0,len(seeds),32)]
 jobs=[(spec,chunk,reference,mode) for mode in modes for spec in (candidate,starter) for chunk in chunks]
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  results=list(pool.map(evaluate_job,jobs))
 report={}
 for index,mode in enumerate(modes):
  size=len(chunks);a=combine(results[(2*index)*size:(2*index+1)*size]);b=combine(results[(2*index+1)*size:(2*index+2)*size])
  report[mode]={'trained':summarize(a),'sdk_starter':summarize(b),'paired_gain':paired_interval(a,b)}
 payload={'model_training':document['training'],'seed_start':seeds[0],'sampling_unit':'four-seat seed block','evaluation':report}
 args.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)


if __name__=='__main__':main()

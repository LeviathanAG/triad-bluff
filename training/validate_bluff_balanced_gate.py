"""Audit raw seed-block evidence without rerunning games or selecting on final seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(document):
    return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def interval(left,right):
    difference=np.asarray(left['block_points'])-np.asarray(right['block_points'])
    rng=np.random.default_rng(20261014)
    bootstrap=difference[rng.integers(0,len(difference),(5000,len(difference)))].mean(axis=1)
    return {'mean_gain':float(difference.mean()),'ci95':np.quantile(bootstrap,[.025,.975]).tolist()}


def close(left,right):
    assert np.allclose(left,right,rtol=0,atol=1e-12),(left,right)


def audit_row(row,summary,count):
    points=np.asarray(row['block_points']);wins=np.asarray(row['block_wins'])
    assert points.shape==wins.shape==(count,)
    assert np.isfinite(points).all() and np.isfinite(wins).all()
    assert (points>=1).all() and (points<=5).all() and (wins>=0).all() and (wins<=1).all()
    assert row['games']==summary['games']==4*count
    close(float(points.mean()),row['points']);close(row['points'],summary['points'])
    close(float(wins.mean()),row['win_rate']);close(row['win_rate'],summary['win_rate'])


def audit_interval(record,actual):
    close(record['mean_gain'],actual['mean_gain']);close(record['ci95'],actual['ci95'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--fit-directory',type=Path,required=True)
    p.add_argument('--linear-backup',type=Path,default=Path('bluffSDK/models/policy-v2.json'))
    p.add_argument('--selection-seed',type=int,default=88000000)
    p.add_argument('--final-seed',type=int,default=89000000)
    p.add_argument('--selection-blocks',type=int,default=128)
    p.add_argument('--final-blocks',type=int,default=256)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    read=lambda path:json.loads(path.read_text())
    selection=read(a.directory/'selection.json');raw=read(a.directory/'selection-blocks.json')
    final=read(a.directory/'evaluation.json');finalraw=read(a.directory/'evaluation-blocks.json')
    fit=read(a.fit_directory/'fit-summary.json')
    docs=[fit['baseline_checkpoint']]+[read(a.fit_directory/f'candidate-{index}.json') for index in range(fit['candidate_count'])]
    hashes=[digest(doc) for doc in docs]
    assert raw['checkpoint_canonical_sha256']==finalraw['checkpoint_canonical_sha256']==hashes
    assert raw['sdk_game_commit']==finalraw['sdk_game_commit']=='96da8042c285af4e4f43655f2ad99592bd49b8e6'
    assert raw['game_config']==finalraw['game_config']=={'players':4,'decks':1,'max_turns':1000,'position_points':[5,3,2,1]}
    assert raw['opponent_pool_canonical_sha256']==finalraw['opponent_pool_canonical_sha256']==digest(selection['opponent_pool'])
    assert raw['seed_ids']==list(range(a.selection_seed,a.selection_seed+a.selection_blocks))
    assert finalraw['seed_ids']==list(range(a.final_seed,a.final_seed+a.final_blocks))
    assert not set(raw['seed_ids'])&set(finalraw['seed_ids'])
    assert selection['seed_start']==a.selection_seed and final['seed_start']==a.final_seed
    expected=['league','neural_selfplay','competitive','truthful','bluffer','weakspot_bluff01_call035','weakspot_bluff065_call035','learned_mixture','strong_rival_selfplay']
    assert selection['selection_regimes']==expected and list(raw['rows'])==expected
    assert selection['max_selection_regression']==.025
    gains={}
    for regime,rows in raw['rows'].items():
        assert len(rows)==len(docs)
        gains[regime]=[]
        for index,row in enumerate(rows):
            audit_row(row,selection['regime_scores'][regime][index],a.selection_blocks)
            if regime=='league':audit_row(row,selection['scores'][index],a.selection_blocks)
            gains[regime].append(row['points']-rows[0]['points'])
            if index:audit_interval(selection['regime_paired_gains'][regime][index-1],interval(row,rows[0]))
    utilities=np.mean(list(gains.values()),axis=0).tolist()
    feasible=[index for index in range(len(docs)) if min(values[index] for values in gains.values())>=-.025]
    chosen=max(feasible,key=lambda index:utilities[index])
    assert chosen==selection['selected']==finalraw['selected_index']
    assert feasible==selection['feasible_candidates'];close(utilities,selection['mean_gain_utilities'])
    selected=read(a.directory/'bluff-policy-mlp.json');assert digest(selected)==hashes[chosen]
    assert finalraw['final_checkpoint_canonical_sha256']['candidate']==hashes[chosen]
    assert finalraw['final_checkpoint_canonical_sha256']['incumbent_neural']==hashes[0]
    assert finalraw['final_checkpoint_canonical_sha256']['retained_v2']==digest(read(a.linear_backup))
    assert len(final['evaluation'])==11
    assert set(finalraw['rows'])==set(final['evaluation'])
    observed={}
    for regime,rows in finalraw['rows'].items():
        report=final['evaluation'][regime]
        for name,row in rows.items():audit_row(row,report[name],a.final_blocks)
        neural=interval(rows['candidate'],rows['incumbent_neural'])
        linear=interval(rows['candidate'],rows['retained_v2'])
        audit_interval(report['paired_gain_vs_neural'],neural);audit_interval(report['paired_gain_vs_v2'],linear)
        observed[regime]=neural
    result={'raw_block_integrity_verified':True,'selected_index':chosen,'selection_utility':utilities[chosen],
            'selection_blocks':a.selection_blocks,'final_blocks':a.final_blocks,'selected_canonical_sha256':hashes[chosen],
            'final_vs_neural_recomputed':observed,'scope':'Evidence audit only; independent fresh gate still required for a proposed replacement.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

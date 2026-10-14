"""Independent path-aware numeric export parity; no model/runtime modification."""
import json,math,sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
from bluff_tree_policy import tree_score
ROOT=Path(__file__).resolve().parent

def reachable_path(tree,row):
    index=0;path=[]
    while True:
        path.append(index);feature,threshold,left,right=tree[index]
        if feature==-1:return path
        index=left if row[feature]<=threshold else right

def choose(lower,upper):
    if lower is None:return min(0.,upper) if upper is not None else 0.
    if upper is None:return max(0.,math.nextafter(lower,math.inf))
    value=lower/2+upper/2
    if not lower<value<=upper:value=math.nextafter(lower,math.inf)
    assert lower<value<=upper
    return value

def original_naive_score(trees,features):
    total=0.
    for tree in trees:
        index=0
        while tree[index][0]!=-1:
            feature,threshold,left,right=tree[index]
            index=left if features[feature]<=threshold else right
        total+=tree[index][1]
    return total

reports=[]
for family in (80,200):
    document=json.loads((ROOT/f'bluff-tree-fit/policy-leaf{family}-strength0.25.json').read_text())
    native=lgb.Booster(model_file=str(ROOT/f'bluff-tree-fit/lightgbm-leaf{family}.txt'))
    probes=[];skipped=0;split_count=0
    for tree in document['trees']:
        stack=[(0,{})]
        while stack:
            index,bounds=stack.pop();feature,threshold,left,right=tree[index]
            if feature==-1:continue
            split_count+=1
            for target in (math.nextafter(threshold,-math.inf),threshold,math.nextafter(threshold,math.inf)):
                lower,upper=bounds.get(feature,(None,None))
                if (lower is not None and target<=lower) or (upper is not None and target>upper):skipped+=1;continue
                row=np.zeros(50,dtype=np.float64)
                for column,(lo,hi) in bounds.items():row[column]=choose(lo,hi)
                row[feature]=target
                assert index in reachable_path(tree,row)
                probes.append(row)
            lo,hi=bounds.get(feature,(None,None))
            left_bounds=dict(bounds);left_bounds[feature]=(lo,min(hi,threshold) if hi is not None else threshold)
            right_bounds=dict(bounds);right_bounds[feature]=(max(lo,threshold) if lo is not None else threshold,hi)
            stack.extend([(left,left_bounds),(right,right_bounds)])
    values=native.predict(np.asarray(probes),num_threads=1)
    plain=np.asarray([tree_score(document['trees'],row) for row in probes])
    error=float(np.max(np.abs(values-plain)));assert error<1e-12
    legacy=np.asarray([original_naive_score(document['trees'],row) for row in probes])
    legacy_error=float(np.max(np.abs(values-legacy)));legacy_mismatches=int(np.count_nonzero(np.abs(values-legacy)>1e-12))
    reports.append({'original_generic_scorer_max_error':legacy_error,'original_generic_mismatch_probes':legacy_mismatches,'family_min_data_in_leaf':family,'splits':split_count,'ancestor_path_confirmed_threshold_probes':len(probes),'infeasible_boundary_probes':skipped,'native_stdlib_max_error':error})
report={'independent_audit':True,'method':'each probe satisfies all ancestor constraints and demonstrably visits the intended numeric split; below/exact/above threshold','families':reports,'scope':'numeric scorer parity only, synthetic vectors are not gameplay observations and do not select a policy'}
(ROOT/'bluff-scorer-path-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

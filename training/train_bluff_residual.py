"""GPU fit of paired rollout preferences; actual-game tests gate improvement."""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bluff_policy_v5 import FEATURE_NAMES
from bluff_mlp_policy import Policy
from train_bluff_league import evaluate_job, paired_interval, summarize


class Residual(nn.Module):
    def __init__(self, base, mean, scale, width):
        super().__init__()
        self.register_buffer('base', torch.tensor(base, dtype=torch.float32))
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        self.hidden = nn.Linear(len(base), width)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, features):
        residual = self.output(F.relu(self.hidden((features - self.mean) / self.scale))).squeeze(-1)
        return features @ self.base + residual, residual


def pairs(data, seed):
    ids = np.unique(data['game_ids']); rng = np.random.default_rng(seed); rng.shuffle(ids)
    validation_ids = set(ids[:max(1, len(ids) // 5)].tolist())
    train, validation, training_rows = [], [], []
    for state, (start, stop) in enumerate(zip(data['offsets'][:-1], data['offsets'][1:])):
        is_validation = int(data['game_ids'][state]) in validation_ids
        destination = validation if is_validation else train
        if not is_validation:
            training_rows.extend(range(start, stop))
        for first in range(start, stop):
            for second in range(first + 1, stop):
                gap = float(data['targets'][first] - data['targets'][second])
                if np.array_equal(data['features'][first], data['features'][second]):
                    continue
                differences = data['trial_targets'][first] - data['trial_targets'][second]
                stderr = float(np.std(differences, ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else 0.
                ranking_weight = min(1., abs(gap) / 3.) * max(.05, 1. - stderr / abs(gap)) if abs(gap) >= .25 else 0.
                # Keep zero-gap observations for value regression. Omitting
                # them would condition each pair on a different nonzero event.
                value_weight = 1. / (1. + stderr ** 2)
                destination.append((first, second, 1. / (1. + np.exp(-gap / .75)), ranking_weight, gap, value_weight))
    if not train or not validation or not any(row[3] > 0. for row in train) or not any(row[3] > 0. for row in validation):
        raise ValueError('not enough informative whole-game-held-out pairs')
    return np.asarray(train), np.asarray(validation), np.asarray(training_rows), len(ids) - len(validation_ids), len(validation_ids)


def loss(model, features, comparisons, penalty, kind):
    a = comparisons[:, 0].long(); b = comparisons[:, 1].long()
    score_a, residual_a = model(features[a]); score_b, residual_b = model(features[b])
    if kind == 'ranking':
        point = F.binary_cross_entropy_with_logits(score_a - score_b, comparisons[:, 2], reduction='none')
        weight = comparisons[:, 3]
    else:
        point = (score_a - score_b - comparisons[:, 4]).square()
        weight = comparisons[:, 5]
    preference = (point * weight).sum() / weight.sum().clamp_min(1e-8)
    return preference + penalty * (residual_a.square().mean() + residual_b.square().mean()) / 2., preference


def export(model, metadata):
    return {'format': 'triad-bluff-residual-mlp-v1', 'features': FEATURE_NAMES,
            'base_weights': model.base.detach().cpu().tolist(),
            'feature_mean': model.mean.detach().cpu().tolist(), 'feature_scale': model.scale.detach().cpu().tolist(),
            'hidden_weight': model.hidden.weight.detach().cpu().tolist(),
            'hidden_bias': model.hidden.bias.detach().cpu().tolist(),
            'output_weight': model.output.weight.detach().cpu().flatten().tolist(),
            'output_bias': float(model.output.bias.detach().cpu()[0]), 'training': metadata}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--validation-blocks', type=int, default=128)
    parser.add_argument('--test-blocks', type=int, default=256)
    parser.add_argument('--validation-seed-start', type=int, default=51_000_000)
    parser.add_argument('--test-seed-start', type=int, default=52_000_000)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--loss-types', default='ranking,value')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    loss_types = args.loss_types.split(',')
    if not loss_types or len(set(loss_types)) != len(loss_types) or any(kind not in ('ranking', 'value') for kind in loss_types):
        parser.error('loss types must be distinct ranking/value choices')
    if min(args.width, args.epochs, args.batch, args.workers, args.validation_blocks, args.test_blocks) < 1 or args.width > 128:
        parser.error('positive counts and width<=128 required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('authorized GPU unavailable; no paid or hardware fallback')
    torch.set_num_threads(1)
    os.environ['BLUFF_FAST_OBSERVATIONS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.data.parent / 'metadata.json').read_text())
    if metadata['input_features'] != FEATURE_NAMES:
        raise ValueError('counterfactual schema differs from deployed feature extractor')
    with np.load(args.data) as archive:
        data = {name: archive[name] for name in archive.files}
    if data['features'].shape[1] != len(FEATURE_NAMES) or not np.isfinite(data['features']).all():
        raise ValueError('invalid training matrix')
    training, validation, training_rows, training_games, validation_games = pairs(data, args.seed)
    mean = data['features'][training_rows].mean(axis=0)
    scale = np.maximum(.1, data['features'][training_rows].std(axis=0))
    # Positive quarter scaling preserves linear move ranking while giving the
    # residual a manageable score scale; only neural parameters are optimized.
    base = [weight / 4. for weight in metadata['incumbent']['weights']]
    base += [0.] * (len(FEATURE_NAMES) - len(base))
    features = torch.as_tensor(data['features'], dtype=torch.float32, device=args.device)
    training = torch.as_tensor(training, dtype=torch.float32, device=args.device)
    validation = torch.as_tensor(validation, dtype=torch.float32, device=args.device)
    print(json.dumps({'device': args.device, 'gpu_count': torch.cuda.device_count(), 'states': len(data['game_ids']),
                      'actions': len(features), 'training_games': training_games, 'validation_games': validation_games,
                      'training_pairs': len(training), 'validation_pairs': len(validation)}), flush=True)
    candidates, history = [], []; started = time.time()
    for index, (loss_type, penalty) in enumerate((kind, penalty) for kind in loss_types for penalty in (.01, .03, .1)):
        torch.manual_seed(args.seed + index)
        model = Residual(base, mean, scale, args.width).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
        with torch.no_grad():
            _, initial_loss = loss(model, features, validation, penalty, loss_type)
        best_loss, best_epoch, best = float(initial_loss), 0, copy.deepcopy(model.state_dict())
        for epoch in range(args.epochs):
            order = torch.randperm(len(training), device=args.device)
            for start in range(0, len(order), args.batch):
                batch = training[order[start:start + args.batch]]
                objective, _ = loss(model, features, batch, penalty, loss_type)
                optimizer.zero_grad(set_to_none=True); objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            with torch.no_grad():
                _, current = loss(model, features, validation, penalty, loss_type)
            if float(current) < best_loss:
                best_loss, best_epoch, best = float(current), epoch + 1, copy.deepcopy(model.state_dict())
            if (epoch + 1) % 10 == 0:
                row = {'loss_type': loss_type, 'regularization': penalty, 'epoch': epoch + 1, 'validation_preference_loss': float(current),
                       'best_preference_loss': best_loss, 'seconds': round(time.time() - started, 1)}
                history.append(row); print(json.dumps(row), flush=True)
        model.load_state_dict(best)
        document = export(model, {'method': 'paired counterfactual residual policy distillation',
            'seed': args.seed + index, 'loss_type': loss_type, 'regularization': penalty, 'best_epoch': best_epoch,
            'validation_preference_loss': best_loss, 'baseline_preference_loss': float(initial_loss),
            'training_games': training_games, 'validation_games': validation_games, 'source_dataset': metadata,
            'base_score_factor': .25, 'hidden_state_use': 'training labels only; policy input is own/public features'})
        check = data['features'][:min(200, len(features))]
        with torch.no_grad():
            tensor = torch.as_tensor(check, dtype=torch.float32, device=args.device)
            _, residual = model(tensor); base_scores = tensor @ model.base
        for strength in (.25, .5, 1.):
            scaled = copy.deepcopy(document)
            scaled['output_weight'] = [weight * strength for weight in scaled['output_weight']]
            scaled['output_bias'] *= strength; scaled['training']['residual_scale'] = strength
            deployed = Policy(scaled)
            actual = np.asarray([deployed.score_features(row.tolist()) for row in check])
            expected = (base_scores + strength * residual).cpu().numpy()
            maximum = float(np.max(np.abs(actual - expected)))
            if maximum > 1e-4:
                raise RuntimeError('standard-library neural export differs from training network')
            path = args.output / f'candidate-{len(candidates)}.json'; path.write_text(json.dumps(scaled, indent=2) + '\n')
            print(json.dumps({'candidate': len(candidates), 'export_max_score_error': maximum}), flush=True)
            candidates.append(scaled)
    (args.output / 'training-progress.json').write_text(json.dumps(history, indent=2) + '\n')
    incumbent = {'kind': 'v2', 'weights': metadata['incumbent']['weights']}
    previous = json.loads(args.previous.read_text())
    reference = [{'kind': kind} for kind in ('starter', 'competitive', 'heuristic', 'random', 'truthful', 'aggressive')]
    reference += [incumbent, {'kind': 'v1', 'weights': previous['weights']}]
    reference += [{'kind': 'bluffer', 'bluff_rate': bluff, 'call_rate': call}
                  for bluff, call in ((.1, .0), (.1, .35), (.35, .1), (.65, .35), (.95, .0), (.95, .35))]
    specs = [incumbent] + [{'kind': 'mlp', 'model': document} for document in candidates]
    selection_seeds = list(range(args.validation_seed_start, args.validation_seed_start + args.validation_blocks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(evaluate_job, [(spec, selection_seeds, reference, 'league') for spec in specs]))
        chosen = max(range(len(results)), key=lambda index: results[index]['points'])
        selection = {'selected': chosen, 'seed_start': args.validation_seed_start,
                     'scores': [summarize(result) for result in results],
                     'paired_gains': [paired_interval(result, results[0]) for result in results[1:]]}
        (args.output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
        print(json.dumps({'actual_game_selection': selection}), flush=True)
        selected = specs[chosen]
        if chosen:
            (args.output / 'bluff-policy-mlp.json').write_text(json.dumps(candidates[chosen - 1], indent=2) + '\n')
        else:
            (args.output / 'retained-incumbent.json').write_text(json.dumps(metadata['incumbent'], indent=2) + '\n')
        final_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_blocks))
        modes = ('league', 'competitive', 'starter', 'truthful', 'bluffer', 'aggressive')
        final = list(pool.map(evaluate_job, [(spec, final_seeds, reference, mode) for mode in modes for spec in (selected, incumbent)]))
        report = {mode: {'candidate': summarize(final[2 * index]), 'incumbent': summarize(final[2 * index + 1]),
                         'paired_gain': paired_interval(final[2 * index], final[2 * index + 1])} for index, mode in enumerate(modes)}
        (args.output / 'evaluation.json').write_text(json.dumps({'seed_start': args.test_seed_start, 'evaluation': report}, indent=2) + '\n')
        print(json.dumps({'held_out_actual_games': report}), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess MCTS jsonl for Offline REINFORCE training.

Input jsonl expected columns:
    messages, expected_acc_reward, answer, task_id, node_id, parent_id, images

Output jsonl adds:
    parent_expected_acc (float)   -- V(s) baseline from parent node
    td_advantage       (float)   -- expected_acc_reward - parent_expected_acc
                                    (i.e. Q(s,a) - V(s))

Filtering logic (see --min_abs_advantage and --drop_dead):
    * dead branch : parent_acc == 0 AND self_acc == 0          -> drop
    * trivial succ: parent_acc == 1 AND self_acc == 1          -> drop
    * |td_advantage| < min_abs_advantage                       -> drop
    * root nodes (parent_id missing)    -> fallback to global mean baseline
"""
import argparse
import collections
import json
import os
import statistics
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--min_abs_advantage', type=float, default=0.05,
                    help='Drop samples with |td_advantage| < this value (default 0.05).')
    ap.add_argument('--drop_dead', type=int, default=1,
                    help='1 = drop dead-branch (parent==0 and self==0) samples.')
    ap.add_argument('--drop_trivial', type=int, default=1,
                    help='1 = drop trivial-success (parent==1 and self==1) samples.')
    ap.add_argument('--fallback', choices=['global_mean', 'zero', 'drop'],
                    default='global_mean',
                    help='How to handle root nodes (no parent in dataset).')
    args = ap.parse_args()

    rows = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f'[info] loaded {len(rows)} rows from {args.input}')

    # Build node_id -> expected_acc_reward map
    node_acc = {}
    for r in rows:
        nid = r.get('node_id')
        if nid is not None:
            node_acc[nid] = float(r.get('expected_acc_reward', 0.0))

    global_mean = statistics.mean(float(r['expected_acc_reward']) for r in rows)
    print(f'[info] global mean expected_acc = {global_mean:.4f}')

    # Compute TD advantage
    kept = []
    drop_stats = collections.Counter()
    td_values = []
    for r in rows:
        self_acc = float(r['expected_acc_reward'])
        pid = r.get('parent_id')
        if pid is not None and pid in node_acc:
            baseline = node_acc[pid]
            baseline_src = 'parent'
        else:
            if args.fallback == 'global_mean':
                baseline = global_mean
                baseline_src = 'global_mean'
            elif args.fallback == 'zero':
                baseline = 0.0
                baseline_src = 'zero'
            elif args.fallback == 'drop':
                drop_stats['no_parent_drop'] += 1
                continue

        td = self_acc - baseline

        # filtering
        if args.drop_dead and baseline == 0.0 and self_acc == 0.0:
            drop_stats['dead_branch'] += 1
            continue
        if args.drop_trivial and baseline == 1.0 and self_acc == 1.0:
            drop_stats['trivial_success'] += 1
            continue
        if abs(td) < args.min_abs_advantage:
            drop_stats['weak_signal'] += 1
            continue

        r_out = dict(r)
        r_out['parent_expected_acc'] = baseline
        r_out['parent_baseline_src'] = baseline_src
        r_out['td_advantage'] = td
        kept.append(r_out)
        td_values.append(td)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # summary
    print(f'\n[done] wrote {len(kept)} rows to {args.output}')
    print(f'[drop stats] {dict(drop_stats)}')
    if td_values:
        print('\n[td_advantage stats]')
        print(f'  mean   : {statistics.mean(td_values):.4f}')
        print(f'  stdev  : {statistics.stdev(td_values) if len(td_values)>1 else 0:.4f}')
        print(f'  min/max: {min(td_values):.4f} / {max(td_values):.4f}')
        pos = sum(1 for v in td_values if v > 0)
        neg = sum(1 for v in td_values if v < 0)
        print(f'  positive: {pos} ({100*pos/len(td_values):.1f}%)')
        print(f'  negative: {neg} ({100*neg/len(td_values):.1f}%)')


if __name__ == '__main__':
    main()

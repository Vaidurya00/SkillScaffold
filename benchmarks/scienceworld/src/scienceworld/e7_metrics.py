import logging
import os
from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step


def log_eval(rollout_id, args, data, extra_metrics=None):
    metrics = dict(extra_metrics or {})
    checkpoint_step = os.environ.get('SCIENCEWORLD_EVAL_CHECKPOINT_STEP')
    metrics['eval/step'] = (
        int(checkpoint_step)
        if checkpoint_step is not None
        else compute_rollout_step(args, rollout_id)
    )
    for name, dataset in data.items():
        samples = list(dataset.get('samples', []))
        groups = {'overall': samples}
        for s in samples:
            groups.setdefault((s.label or {}).get('task_name', 'unknown'), []).append(s)
        for task, samples in groups.items():
            if not samples: continue
            p = f'eval/{name}/{task}'
            task_keys = {
                (
                    (s.label or {}).get('task_name'),
                    (s.label or {}).get('variation_idx'),
                    (s.label or {}).get('split'),
                )
                for s in samples
            }
            metrics[p+'/count'] = len(task_keys)
            metrics[p+'/trajectory_count'] = len(samples)
            metrics[p+'/pass_at_1'] = sum(bool((s.metadata or {}).get('won')) for s in samples)/len(samples)
            won_by_task = {}
            for s in samples:
                label = s.label or {}
                key = (
                    label.get('task_name'),
                    label.get('variation_idx'),
                    label.get('split'),
                )
                won_by_task.setdefault(key, []).append(bool((s.metadata or {}).get('won')))
            metrics[p+'/pass_at_k'] = sum(any(values) for values in won_by_task.values())/len(won_by_task)
            metrics[p+'/task_score'] = sum(float((s.metadata or {}).get('task_score', 0)) for s in samples)/len(samples)
            metrics[p+'/invalid_count'] = sum('task_score' not in (s.metadata or {}) or bool((s.metadata or {}).get('infra_error')) for s in samples)
            metrics[p+'/truncated_ratio'] = sum(str(s.status).lower().endswith('truncated') for s in samples)/len(samples)
    logging.getLogger(__name__).info('SCIENCEWORLD_E7_EVAL %s', metrics)
    logging_utils.log(args, metrics, step_key='eval/step')
    return True

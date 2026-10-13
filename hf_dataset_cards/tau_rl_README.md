---
license: apache-2.0
language:
- en
task_categories:
- reinforcement-learning
tags:
- tool-use
- agent
- multi-turn
- tau2-bench
pretty_name: SkillScaffold Tau RL Training Data
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
---

# SkillScaffold Tau RL Training Data

This repository contains the three-domain Tau RL task pool used by
SkillScaffold. It follows the release layout and task representation of
[inclusionAI/AReaL-tau2-data](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data):
each JSONL row is a complete task and references an environment snapshot using
the repository-relative `db_path` field.

## Contents

```text
train.jsonl
tau2_rl_database/
  tau2_airline_db.json
  tau2_airline_new_db_1.json
  tau2_airline_new_db_2.json
  tau2_airline_new_db_3.json
  tau2_retail_new_db_1.json
  tau2_retail_new_db_2.json
  tau2_retail_new_db_3.json
  tau2_retail_new_db_4.json
  tau2_telecom_db.toml
```

The dataset contains 900 training tasks, balanced across three domains.

## Data sources and selection

- **Airline and Retail:** Candidate tasks are drawn from
  [AReaL-tau2-data](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data).
  We evaluate each task with Qwen3-8B using eight rollouts and retain tasks
  solved in 1–6 out of 8 rollouts. We then construct a 300-task training pool
  for each domain. Retail tasks are randomly sampled from the eligible pool;
  Airline tasks use distribution-aware resampling to better match the target
  task-family and conservative-operation distribution.

- **Telecom:** Candidate tasks are drawn from the non-base training pool of
  [Tau2](https://github.com/sierra-research/tau2-bench), excluding evaluation
  tasks. We apply the same Qwen3-8B screening rule, retaining tasks solved in
  1–6 out of 8 rollouts, and randomly sample 300 eligible tasks.

## Data format

Each row has the same core fields as the upstream AReaL Tau RL release:

```json
{
  "id": "...",
  "description": {...},
  "user_scenario": {...},
  "evaluation_criteria": "{...}",
  "db_path": "tau2_rl_database/..."
}
```

Telecom tasks may additionally contain `ticket` and `initial_state`.

## Acknowledgements

We sincerely thank the authors of
[AReaL-tau2-data](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data)
for releasing the synthetic Airline and Retail training tasks and environment
database snapshots. We also thank the authors of
[Tau2](https://github.com/sierra-research/tau2-bench) for releasing the
Telecom tasks, benchmark, and environments.

The nine database snapshots are redistributed unchanged. Please cite and
acknowledge both the AReaL/SEA and Tau2 projects when using this dataset.

## Upstream citations

```bibtex
@article{gao2025sea,
  title={From Self-Evolving Synthetic Data to Verifiable-Reward RL:
         Post-Training Multi-turn Interactive Tool-Using Agents},
  author={Gao, Jiaxuan and Chen, Jiaao and He, Chuyi and
          Wang, Wei-Chen and Xu, Shusheng and Wang, Hanrui and
          Jin, Di and Wu, Yi},
  journal={arXiv preprint arXiv:2601.22607},
  year={2025}
}

@article{fu2025areal,
  title={AReaL: A Large-Scale Asynchronous Reinforcement Learning
         System for Language Reasoning},
  author={Fu, Wei and Gao, Jiaxuan and Shen, Xujie and Zhu, Chen and
          Mei, Zhiyu and He, Chuyi and Xu, Shusheng and Wei, Guo and
          Mei, Jun and Wang, Jiashu and Yang, Tongkai and
          Yuan, Binhang and Wu, Yi},
  journal={arXiv preprint arXiv:2505.24298},
  year={2025}
}

@misc{barres2025tau2,
  title={$\tau^2$-Bench: Evaluating Conversational Agents
         in a Dual-Control Environment},
  author={Victor Barres and Honghua Dong and Soham Ray and
          Xujie Si and Karthik Narasimhan},
  year={2025},
  eprint={2506.07982},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

## License

The upstream AReaL dataset is released under Apache-2.0, and the Tau2
repository is released under MIT. Users must also comply with the terms of any
downstream training framework.

# Training on Kaggle

Notebook settings: **Accelerator: GPU T4 x2**, **Internet: on**, **Persistence: off**.
Add the datasets as inputs: your `pusht_human_*` / `pusht_random` dataset, and on later
sessions the previous session's output (for resuming).

If you clone a private fork, store a GitHub token (fine-grained, read access) under
**Add-ons -> Secrets** as `GITHUB_TOKEN`; a public clone needs no token.

## Cells

```python
# 1. code
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone -q --branch phone2panda-fixes https://{tok}@github.com/05kashyap/humanai-ts /kaggle/working/humanai-ts
%cd /kaggle/working/humanai-ts
```

```python
# 2. training dependencies (a few minutes)
!bash kaggle/kaggle_setup.sh
```

```python
# 3. train two runs at once, one per T4 (check the input path in the right-hand panel)
!DATASET_DIR=/kaggle/input/pusht-human EPOCHS=10 bash kaggle/kaggle_train.sh aug random
```

Watch a run while it trains:

```python
!tail -n 3 /kaggle/working/checkpoints/logs/humanai_aug.log
```

## Sessions, resuming, downloading

- Use **Save Version -> Save & Run All (Commit)** for long runs: it runs in the background
  for up to 12 h and keeps everything in `/kaggle/working` as the version's output.
- To continue a run in a new session, add that output as an input and pass it:
  `RESUME_DIR=/kaggle/input/<previous-output> ... kaggle_train.sh aug random`.
  Each run resumes from its last checkpoint (one every 1000 iterations) up to `EPOCHS`.
- The checkpoints are in `checkpoints/test/humanai_<run>/` of the output; download them
  for planning.

## Memory

Defaults are batch 16 (32 runs out of memory on a T4) with the decoder on. If a run runs
out of GPU memory, add `DECODER=0` (the decoder only adds images).

## Budget

Kaggle gives about 30 GPU-hours a week. Check the speed in the first log lines
(iterations per second; one epoch of a 125k-step dataset is about 6000 iterations at
batch 16) and pick `EPOCHS` so the runs you need fit. Priority: `aug random`, then
`noreg`, then `branch raw`.

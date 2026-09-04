# patches/

Diffs against the upstream clones, one file per experiment, so a change is
reproducible even though `tron1-rl-isaaclab/` itself is not tracked here.

Export:

```bash
cd ~/rc_lab/study_tron1/tron1-rl-isaaclab
git diff > ../patches/exp_003_energy_weight.diff
```

Apply:

```bash
cd ~/rc_lab/study_tron1/tron1-rl-isaaclab
git apply ../patches/exp_003_energy_weight.diff
```

Always record the base commit in the experiment note (`git rev-parse --short HEAD`),
otherwise the diff may not apply later.

> Once the upstream repo is forked, prefer a branch per experiment in the fork and
> keep only the branch name in the experiment note. Patches are the fallback for
> quick throwaway edits.

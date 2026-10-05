# Conformal Safety Margins

Code for collecting the reduction datasets and reproducing the figures of *Conformal Safety Margins for Reinforcement Learning*.

## Install

SafetyPointGoal1 needs its own environment, because safety-gymnasium pins an older gymnasium.

```bash
# CartPole, Pendulum, BeamRider, grouping, figures (Python >= 3.10)
pip install -r requirements.txt
# SafetyPointGoal1 (Python 3.8)
pip install -r requirements-safety.txt
```

Run every command from the repository root. Outputs are written to `runs/`.

## SafetyPointGoal1

Train the CPO policy with [SafePO](https://github.com/PKU-Alignment/Safe-Policy-Optimization), from a clone of that repository:

```bash
python safepo/single_agent/cpo.py --task SafetyPointGoal1-v0 --seed 0 --total-steps 12000000 --num-envs 10 --steps-per-epoch 20000 --cost-limit 25
```

Set `CPO_RUN` to the resulting run directory, which contains `config.json`, `torch_save/` and `state*.pkl`.

```bash
python spg1/safety_margin_grushin.py --eval-dir $CPO_RUN --num-anchors 6000 --samples-per-state 48 --tolerance-list 0.2 0.3 0.4 0.5 0.6 0.8 1.0 1.2 1.5 2.0 2.5 3.0 --raw-npz runs/spg1/sweep/raw.npz --save-path runs/spg1/sweep/grushin_net.pkl --results-json runs/spg1/sweep/results.json
python spg1/safety_margin_adaptivity_direct.py --eval-dir $CPO_RUN --num-anchors 10000 --samples-per-state 64 --random-steps 16 --tau-list 0.8 1.2 1.7 --mondrian-groups 3 --calibration-fraction 0.3 --raw-npz runs/spg1/n16/raw.npz --results-json runs/spg1/n16/results.json
```

## CartPole and Pendulum

The pretrained `sb3/ppo-*` agents are downloaded from the Hugging Face Hub.

```bash
python cartpole/cartpole_grushin.py --num-anchors 5000 --samples-per-state 64 --tolerance-quantiles 0.5 0.7 0.85 0.95 --raw-npz runs/cartpole/sweep/raw.npz --save-path runs/cartpole/sweep/grushin_net.pkl --results-json runs/cartpole/sweep/results.json
python cartpole/cartpole_adaptivity_direct.py --num-anchors 44000 --samples-per-state 96 --random-steps 8 --no-terminal-fill --tau-quantiles 0.5 0.7 0.85 0.95 --raw-npz runs/cartpole/n8/raw.npz --results-json runs/cartpole/n8/results.json
python pendulum/pendulum_grushin.py --num-anchors 15000 --samples-per-state 96 --tolerance-list 5 10 25 50 100 150 --raw-npz runs/pendulum/sweep/raw.npz --save-path runs/pendulum/sweep/grushin_net.pkl --results-json runs/pendulum/sweep/results.json
python pendulum/pendulum_adaptivity_direct.py --num-anchors 5000 --samples-per-state 64 --random-steps 8 --tau-list 0 5 100 --raw-npz runs/pendulum/n8/raw.npz --results-json runs/pendulum/n8/results.json
```

## BeamRider

This uses the pretrained `sb3/qrdqn-BeamRiderNoFrameskip-v4` agent. The second command runs the density-estimate baseline of Grushin et al. on the same data. The third command analyses the n = 8 slice of that data.

```bash
python atari/atari_grushin.py --num-anchors 5000 --samples-per-state 128 --random-prob 0.02 --gamma 0.999 --max-steps 400 --episodic-life --tolerance-list 5 10 15 20 25 30 40 50 75 100 150 200 250 300 400 500 650 800 1000 --raw-npz runs/beamrider/sweep/raw.npz --save-path runs/beamrider/sweep/grushin_net.pkl --results-json runs/beamrider/sweep/results.json
python atari/atari_grushin_kde.py --raw-npz runs/beamrider/sweep/raw.npz --tolerance-list 25 75 150 300 --results-json runs/beamrider/kde/results.json
python atari/atari_adaptivity_direct.py --sweep-npz runs/beamrider/sweep/raw.npz --sweep-n 8 --tau-quantiles 0.7 0.85 0.95 --mondrian-groups 6 --results-json runs/beamrider/n8/results.json
```

## Group-conditional analysis and figures

Run `export spg1` in the SafetyPointGoal1 environment and everything else in the main environment.

```bash
python grouping/grouped_sweep.py export spg1
python grouping/grouped_sweep.py export cartpole pendulum beamrider
python grouping/fixed_groups.py all spg1 cartpole pendulum beamrider
python grouping/fixed_agree.py spg1 cartpole pendulum beamrider
python figures/make_figures.py
python figures/make_method_figure.py
```

## Common options

Every experiment script also accepts these:

| Option | Meaning |
|---|---|
| `--num-workers` | number of parallel rollout workers |
| `--seed` | random seed (0 in the paper) |
| `--alpha` | significance level (0.1 in the paper) |
| `--plot-dir` | write diagnostic plots |
| `--n-list` | perturbation lengths for the sweep scripts (default `1 2 4 8 16 32`) |

`python <script> --help` lists all options.

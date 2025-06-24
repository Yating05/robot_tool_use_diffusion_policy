# using images
python train.py --config-dir=. --config-name=victor_diffusion_policy.yaml training.seed=7 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# without using images
python train.py --config-dir=. --config-name=victor_diffusion_policy_state.yaml training.seed=7 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
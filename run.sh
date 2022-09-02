mkdir -p results/trainlogs

critic=$1
seed=$2
gpu=$3

lr=0.001

env=FetchReach
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 25  --agent her --negative-reward --critic $critic --seed $seed --cuda
env=FetchPush
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 50  --agent her --negative-reward --critic $critic --seed $seed --cuda
env=FetchSlide
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 50  --agent her --negative-reward --critic $critic --seed $seed --cuda
env=FetchPick
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 50  --agent her --negative-reward --critic $critic --seed $seed --cuda
env=HandManipulateBlockRotateZ
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 50  --agent her --negative-reward --critic $critic --seed $seed --cuda
env=HandManipulateBlockRotateParallel
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 100 --agent her --negative-reward --critic $critic --seed $seed --cuda
env=HandManipulateBlockRotateXYZ
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 100 --agent her --negative-reward --critic $critic --seed $seed --cuda
env=HandManipulateBlockFull
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 100 --agent her --negative-reward --critic $critic --seed $seed --cuda
env=HandManipulateEggRotate
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 50  --agent her --negative-reward --critic $critic --seed $seed --cuda
env=HandManipulateEggFull
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 100 --agent her --negative-reward --critic $critic --seed $seed --cuda
env=HandManipulatePenRotate
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 50  --agent her --negative-reward --critic $critic --seed $seed --cuda
env=HandManipulatePenFull
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs 100 --agent her --negative-reward --critic $critic --seed $seed --cuda

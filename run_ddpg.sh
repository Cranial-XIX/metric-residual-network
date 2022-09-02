mkdir -p results/trainlogs

seed=$1
gpu=$2

env=FetchReach
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 25  --agent ddpg --negative-reward --seed $seed --cuda
env=FetchPush
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 50  --agent ddpg --negative-reward --seed $seed --cuda
env=FetchSlide
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 50  --agent ddpg --negative-reward --seed $seed --cuda
env=FetchPick
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 50  --agent ddpg --negative-reward --seed $seed --cuda
env=HandManipulateBlockRotateZ
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 50  --agent ddpg --negative-reward --seed $seed --cuda
env=HandManipulateBlockRotateParallel
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 100 --agent ddpg --negative-reward --seed $seed --cuda
env=HandManipulateBlockRotateXYZ
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 100 --agent ddpg --negative-reward --seed $seed --cuda
env=HandManipulateBlockFull
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 100 --agent ddpg --negative-reward --seed $seed --cuda
env=HandManipulateEggRotate
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 50  --agent ddpg --negative-reward --seed $seed --cuda
env=HandManipulateEggFull
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 100 --agent ddpg --negative-reward --seed $seed --cuda
env=HandManipulatePenRotate
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 50  --agent ddpg --negative-reward --seed $seed --cuda
env=HandManipulatePenFull
export CUDA_VISIBLE_DEVICES=$gpu && python -u main.py --env-name=$env --n-epochs 100 --agent ddpg --negative-reward --seed $seed --cuda

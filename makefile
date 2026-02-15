# ALGO ?= dicg_ctde 
ALGO ?= vast 

all: 
	CUDA_VISIBLE_DEVICES=0 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-adversary-v3" env_args.pretrained_wrapper="PretrainedAdversary" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=0 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-speaker-listener-v4" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=0 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-world-comm-v3" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=1 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-crypto-v3" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=1 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=2 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-push-v3" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=2 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=3 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-reference-v3" seed=10 t_max=25000 use_cuda=True &
 
test-on-cpu: 
	clear 
# 	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="matrixgames:penalty-100-nostate-v0" seed=10 t_max=25000 use_cuda=False 
# 	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3" seed=10 t_max=25000 use_cuda=False 
# 	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v2" seed=10 t_max=25000 use_cuda=False 
	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" seed=10 t_max=25000 use_cuda=False
# 	python src/main.py --config=$(ALGO) --env-config=sc2 with env_args.map_name="3s5z" seed=10 t_max=25000 use_cuda=False 
# 	python src/main.py --config=$(ALGO) --env-config=sc2v2 with env_args.map_name="protoss_5_vs_5" seed=10 t_max=25000 use_cuda=False 
# 	python src/main.py --config=$(ALGO) --env-config=smaclite with env_args.time_limit=150 env_args.map_name="MMM" seed=10 t_max=25000 use_cuda=False 
# 	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=150 env_args.key="vmas-balance" seed=10 t_max=25000 use_cuda=False 

	
	python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=hallway seed=10 t_max=25000 use_cuda=False
# 	python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=gather seed=10 t_max=25000 use_cuda=False 
	python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=disperse seed=10 t_max=25000 use_cuda=False 
# 	python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=pursuit seed=10 t_max=25000 use_cuda=False 
# 	python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=aloha seed=10 t_max=25000 use_cuda=False 
# 	python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=sensor seed=10 t_max=25000 use_cuda=False 

test-on-cpu-mpe:
	clear
	python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=hallway seed=10 t_max=25000 use_cuda=False 
	python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=disperse seed=10 t_max=25000 use_cuda=False 
	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-reference-v3" seed=10 t_max=25000 use_cuda=False use_wandb=True
	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-speaker-listener-v4" seed=10 t_max=25000 use_cuda=False use_wandb=True
	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-crypto-v3" seed=10 t_max=25000 use_cuda=False use_wandb=True
	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-adversary-v3" env_args.pretrained_wrapper="PretrainedAdversary" seed=10 t_max=25000 use_cuda=False use_wandb=True

# 	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-push-v3" seed=10 t_max=25000 use_cuda=False use_wandb=True
# 	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" seed=10 t_max=25000 use_cuda=False use_wandb=True
# 	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" seed=10 t_max=25000 use_cuda=False use_wandb=True
# 	python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-world-comm-v3" seed=10 t_max=25000 use_cuda=False use_wandb=True

test-on-gpu: 
	clear 
	CUDA_VISIBLE_DEVICES=0 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="matrixgames:penalty-100-nostate-v0" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=1 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=2 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v2" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=3 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=4 python src/main.py --config=$(ALGO) --env-config=sc2 with env_args.map_name="3s5z" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=5 python src/main.py --config=$(ALGO) --env-config=sc2v2 with env_args.map_name="protoss_5_vs_5" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=6 python src/main.py --config=$(ALGO) --env-config=smaclite with env_args.time_limit=150 env_args.map_name="MMM" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=7 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=0 python src/main.py --config=$(ALGO) --env-config=gymma with env_args.time_limit=150 env_args.key="vmas-balance" seed=10 t_max=25000 use_cuda=True &
	
	CUDA_VISIBLE_DEVICES=1 python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=hallway seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=2 python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=gather seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=3 python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=disperse seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=4 python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=pursuit seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=5 python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=aloha seed=10 t_max=25000 use_cuda=True &
	CUDA_VISIBLE_DEVICES=6 python src/main.py --config=$(ALGO) --env-config=maco with env_args.map_name=sensor seed=10 t_max=25000 use_cuda=True &

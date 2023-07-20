CUDA_VISIBLE_DEVICES=1 python main.py --datname cora --agg NR --att 0.66 --dropout 0.061 --l1 0.46 --lap 0.9 --lr 0.025 --nbsz 20 --ncaps 12 --nhidden 52 --nlayer 7 --reg 0.003 --routit 6 --iterat 8

CUDA_VISIBLE_DEVICES=4 python main.py --datname citeseer --agg NR --att 0.1 --dropout 0.39 --l1 0.9 --lap 0.4 --lr 0.01 --nbsz 20 --ncaps 12 --nhidden 40 --nlayer 3 --reg 0.0034 --routit 6

CUDA_VISIBLE_DEVICES=2 python main.py --datname pubmed --agg NR --att 0.062 --dropout 0.17 --l1 0.49 --lap 1 --lr 0.064 --nbsz 20 --ncaps 12 --nhidden 36 --nlayer 6 --reg 9.4e-05 --routit 6 --iterat 8

CUDA_VISIBLE_DEVICES=3 python main.py --datname amazon_electronics_photo.npz --att 0.2 --dropout 0.055 --l1 0.29 --lap 0.68 --lr 0.012 --nbsz 20 --ncaps 12 --nhidden 52 --nlayer 1 --reg 0.00012 --routit 6 --iterat 8

CUDA_VISIBLE_DEVICES=5 python main.py --datname chameleon.npz --agg MEAN --att 0.4 --dropout 0.013 --l1 0.1 --lap 0.5 --lr 0.025 --nbsz 20 --ncaps 12 --nhidden 56 --nlayer 1 --reg 4.8e-05 --routit 6

CUDA_VISIBLE_DEVICES=4 python main.py --datname squirrel.npz --agg MEAN --att 0.25 --dropout 0.015 --l1 0.4 --lap 0.25 --lr 0.0075 --nbsz 20 --ncaps 12 --nhidden 56 --nlayer 1 --reg 6.3e-05 --routit 6

CUDA_VISIBLE_DEVICES=4 python main.py --datname crocodile.npz --agg MEAN --att 0.1 --dropout 0.076 --l1 0.9 --lap 0.9 --lr 0.025 --nbsz 20 --ncaps 12 --nhidden 48 --nlayer 1 --reg 7.4e-05 --routit 6
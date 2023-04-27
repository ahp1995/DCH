# DCH
Dataset Condensation with High Frequency Detail

The Baseline code from https://github.com/VICO-UoE/DatasetCondensation

python main.py --dataset CIFAR100 --model ConvNet --batch_real 128 --batch_train 128 --ipc 10 \
--init real --num_exp 1 --num_eval 1 --eval_mode M --Iteration 500 --filter dogf --hpf_param 0.01

dir_name='linear_vanilla_1e-2'
model='LinearMetricNet'
info='vanilla'
n_layers=16
n_hidden=50
in_features=3
out_features=6890
embedding_dim=3
n_epochs=2000
batch_size=1
seed=1
scheduler='cosine'
lr=1e-2
min_lr=1e-3
tikhonov=0

python train/correspondence.py \
    --seed ${seed} \
    --dir-name ${dir_name} \
    --model ${model} \
    --info ${info} \
    --in-features ${in_features} \
    --out-features ${out_features} \
    --n-layers ${n_layers} \
    --n-hidden ${n_hidden} \
    --embedding-dim ${embedding_dim} \
    --n-epochs ${n_epochs} \
    --batch-size ${batch_size} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --scheduler ${scheduler} \
    --tikhonov ${tikhonov} 


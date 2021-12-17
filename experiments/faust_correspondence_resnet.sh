dir_name='crap'
model='MetricResNet'
info='face'
n_layers=8
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
metric_penalty=1e-6

python train/correspondence.py \
    --seed ${seed} \
    --dir-name ${dir_name} \
    --model ${model} \
    --dataset 'FAUST' \
    --info ${info} \
    --in-features ${in_features} \
    --out-features ${out_features} \
    --n-layers ${n_layers} \
    --n-hidden ${n_hidden} \
    --embedding-dim ${embedding_dim} \
    --n-epochs ${n_epochs} \
    --batch-size ${batch_size} \
    --scheduler ${scheduler} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --tikhonov ${tikhonov} \
    --metric-penalty ${metric_penalty}

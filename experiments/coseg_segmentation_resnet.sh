dir_name='crap'
dataset='COSEG'
classes="chairs"
model='MetricResNet'
info='face'
in_features=3
out_features=4
n_layers=8
n_hidden=100
embedding_dim=10
n_epochs=750
batch_size=1
seed=1
lr=1e-2
min_lr=1e-4
scheduler='cosine'
metric_penalty=1e-3

python train/segmentation.py \
    --seed ${seed} \
    --dir-name ${dir_name} \
    --dataset ${dataset} \
    --classes ${classes} \
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
    --metric-penalty ${metric_penalty}


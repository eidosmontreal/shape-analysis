root='/home/babbasi/Perforce/babbasi_MTLWKS1562a_271/AI/ML/StyleTransfer/shape_analysis/logs/HumanSegmentation1000/segmentation/face_feat_1'
n_samples=30

python scripts/demo_segmentation.py \
    --root ${root} \
    --n-samples ${n_samples}

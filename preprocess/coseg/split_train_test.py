import os, subprocess, sys
from os.path import join

"""
Splits the COSEG data into a training and testing set with the set ``ratio``.
The script is to be used with ``downlod_coseg.sh`` to download and process the datatset.
"""
assert len(sys.argv) > 1, "Please specify data directory."

#root = join('..','data','COSEG') # Data directory
root=sys.argv[1]
ratio = float(sys.argv[2]) # Perentage of training data 
for category in ['chairs','vases','tele_aliens']:
    for mode in ['train','test']:
        for data in ['gt','shapes']:
            subprocess.call(['mkdir -p {}'.format(join(root,mode,category,data))],shell=True)
        
    files = os.listdir(join(root,category,'shapes'))
    num_files = len(files)
    for i,f in enumerate(files):
        g = f.split('.')[0]+'.seg'

        old_shape_path = join(root,category,'shapes',f)
        old_gt_path = join(root,category,'gt',g)
       
        mode = 'train' if (i+1)/num_files <= ratio else 'test'

        new_shape_path = join(root,mode,category,'shapes',f)
        new_gt_path = join(root,mode,category,'gt',g)
        
        subprocess.call(['cp {} {}'.format(old_shape_path,new_shape_path)],shell=True)
        subprocess.call(['cp {} {}'.format(old_gt_path,new_gt_path)],shell=True)




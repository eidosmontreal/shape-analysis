import subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, required=True, help='Directory where data is stored')
parser.add_argument("--ratio", type=float, default=0.85, required=False, help='Ratio of train to test data')
args = parser.parse_args()

subprocess.call(['bash coseg/download_data.sh {}'.format(args.data_dir)],shell=True)
print('Splitting test and train data...')
subprocess.call(['python coseg/split_train_test.py {} {}'.format(args.data_dir, args.ratio)],shell=True)
print('Converting face labels to vertex labels...')
subprocess.call(['python coseg/convert_labels.py {}'.format(args.data_dir)],shell=True)
print('...done!')


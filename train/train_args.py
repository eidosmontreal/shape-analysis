import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    
    # Experiment parameters
    parser.add_argument("--dir-name", type=str, default=None, help="Name of directory to store training data.")
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for training.")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, choices=['FAUST','COSEG','HumanSegmentation'], help="Dataset to train on.", required=True)
    parser.add_argument("--classes", nargs='*', default=['chairs'],help="Classes to train on for COSEG.")
    parser.add_argument("--num-vertices", type=int, default=None, help="Number of vertices per mesh.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size of training iterations.")
   
    # Model parameters
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--classification", action='store_true', help="Indicate whether task is a classification task or not.")
    parser.add_argument("--info", type=str, default='face', help="Type of MetricConv to use")
    parser.add_argument("--embedding-dim", type=int, default=3, help="Embedding dimension of MetricConv.")
    parser.add_argument("--in-features", type=int, default=3, help="Number of input features required for model.")
    parser.add_argument("--out-features", type=int, default=8, help="Number of output features returned by model.")
    parser.add_argument("--n-hidden", type=int, default=100, help="Size of hidden layer. Total number of features = n_hidden**2.")
    parser.add_argument("--n-layers", type=int, default=10, help="Number of layers with n_hidden features.")
    parser.add_argument("--remove-reference",action='store_true', help="Removes reference node from message-passing.")
    parser.add_argument("--symmetric",action='store_true',help="Uses (A+A^T)/2 as message-passing.")
   
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for optimizer")
    parser.add_argument("--scheduler", type=str, default=None, choices=['cosine','step'],
                                        help="Scheduler for optimizer's learning rate.")
    parser.add_argument("--min-lr", type=float, default=0, help="Minimum learning rate for scheduler")
    parser.add_argument("--decay-freq", type=int, default=1, help="Frequency of reducing LR.")
    parser.add_argument("--decay-rate", type=float, default=0.99, help="Decay rate of LR for step scheduler.")
    parser.add_argument("--tikhonov", type=float, default=0., help="Weight decay/tikhonov weight for optimizer.")
    parser.add_argument("--metric-penalty", type=float, default=0, help="Weight of metric tensor penalty.")
    
    return parser
    

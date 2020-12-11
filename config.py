import argparse

def configuration():
    parser = argparse.ArgumentParser()
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, default='../train', help='Dataset path. Codes will recursively find numpy files from root directory')
    parser.add_argument('--domain_A', type=str, default='LDCT', help="Domain name for A. utils.get_data function will find data path containing this argument")
    parser.add_argument('--domain_B', type=str, default='SDCT', help="Domain name for B. utils.get_data function will find data path containing this argument")
    parser.add_argument('--extension', type=str, default='npy', help="File extension")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='./results')
    
    # Training arguments
    parser.add_argument('--gpu', type=str, default='3', help="Numbers of GPU that will be used")
    
    # Model arguments
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--channel', type=int, default=1, help="Number of channels of input image")
    parser.add_argument('--n_feature', type=int, default=64, help="Number of feature maps")
    parser.add_argument('--factor', type=int, default=2, help="Rate for feature map increase")
    parser.add_argument('--D_kernel_size', type=int, default=4, help='Number of kernel size for discriminator')
    parser.add_argument('--D_stride', type=int, default=2, help='Number for convolution stride for discriminator')
    parser.add_argument('--image_size', type=int, default=512, help="Input image size")
    parser.add_argument('--G_lr', type=float, default=0.00001)
    parser.add_argument('--D_lr', type=float, default=0.00001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eta', type=float, default=0.5, help="Regularization hyperparameter for gradient penalty")
    parser.add_argument('--gamma', type=float, default=5, help="Hyperparameter for cyclic loss")
    parser.add_argument('--normalize', type=str, default='CT', help='Normalization method for [numpy array]/[torch tensor array]', choices=['minmax', 'tanh', 'CT', 'None'])
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--G_iter', type=int, default=6000, help="Number of iteration for generator")
    parser.add_argument('--D_iter', type=int, default=300, help="Number of iteration for discriminator")
    parser.add_argument('--D_max_iter', type=int, default=6000, help="Max iteration for discriminator")
    parser.add_argument('--lr_decay', type=str, default='linear', help="Learing rate decay strategy")
    parser.add_argument('--decay_epoch', type=int, default=30, help="Learning rate decay start epoch")
    
    # Visualize arguments
    parser.add_argument('--print_step', type=int, default=100, help='Steps for intermediate result print')
    
    return parser.parse_args()
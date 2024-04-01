import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')

    parser.add_argument('--amp', action='store_true',
                        help='specify if using amp in training')

    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--local_rank',
                        default=-1,
                        type=int,
                        help='the local rank for distributed training. ')

    parser.add_argument('--student_config_file', type=str, default='configs/models/vitb32_large_language.yaml',
                        help='the base path of datasets')

    # ####################  dataset #########################
    parser.add_argument('--dataroot', type=str, default='/fsx/sunxm/datasets/',
                        help='the base path of datasets')
    parser.add_argument('--tsv_file_list', type=str, nargs='+',  default=[],
                        help='the large scale dataset')
    parser.add_argument('--batch_size', type=int, default=640,
                        help='batch size used for the  dataset')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='the number of data workers')


    # ########################### optimizers and schedulers ###########################
    parser.add_argument('--base_lr', type=float, default=8e-4, help='the base learning rate')
    parser.add_argument('--warmup_lr', type=float, default=4e-6, help='the warmup learning rate')
    parser.add_argument('--min_lr', type=float, default=4e-5, help='the min learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='the number of epochs to train image encoder')
    parser.add_argu
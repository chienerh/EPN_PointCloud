from SPConvNets.trainer_oxford import Trainer
from SPConvNets.options import opt as opt_oxford
# optional, wandb
import wandb


if __name__ == '__main__':
    
    opt_oxford.batch_size = 1
    opt_oxford.num_iterations = 16001
    # opt_oxford.num_epochs = 30
    opt_oxford.save_freq = 4000
    opt_oxford.train_lr.decay_step = 20000
    opt_oxford.no_augmentation = True # TODO
    # opt_oxford.train_file = '/home/cel/data/benchmark_datasets/training_queries_baseline_3seq.pickle'
    # opt_oxford.val_file = '/home/cel/data/benchmark_datasets/test_queries_baseline_3seq.pickle'
    opt_oxford.train_file = '/home/cel/data/benchmark_datasets/training_queries_tran_noise.pickle'
    opt_oxford.val_file = '/home/cel/data/benchmark_datasets/test_queries_tran_noise.pickle'
    # opt_oxford.train_file = '/home/cel/data/benchmark_datasets/training_queries_baseline_seq567.pickle'
    # opt_oxford.val_file = '/home/cel/data/benchmark_datasets/test_queries_baseline_seq567.pickle'
    opt_oxford.npt = 4096 # ???  
    opt_oxford.model.model = 'pr_so3net_netvlad' 

    # IO
    opt_oxford.model.input_num = 4096
    opt_oxford.model.output_num = 64
    opt_oxford.global_feature_dim = 128

    # place recognition
    opt_oxford.num_points = 4096
    opt_oxford.num_selected_points = 128 #150
    opt_oxford.pos_per_query = 2
    opt_oxford.neg_per_query = 6

    # loss
    opt_oxford.LOSS_FUNCTION = 'quadruplet'
    opt_oxford.MARGIN_1 = 0.5
    opt_oxford.MARGIN_2 = 0.2
    opt_oxford.LOSS_LAZY = True
    opt_oxford.LOSS_IGNORE_ZERO_BATCH = False
    opt_oxford.TRIPLET_USE_BEST_POSITIVES = False

    # optional, wandb
    wandb.init(
        project='train_epn_netvlad_oxford_tran_noise',
        config={
        "architecture": 'pr_so3net_netvlad',
        "dataset": 'oxford',})

    trainer = Trainer(opt_oxford)
    if opt_oxford.mode == 'train':
        trainer.train()
    elif opt_oxford.mode == 'eval':
        trainer.eval() 

    # optional, wandb
    wandb.finish()
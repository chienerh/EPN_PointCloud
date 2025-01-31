from SPConvNets.trainer_modelnet import Trainer
from SPConvNets.options import opt
# optional, wandb
import wandb

if __name__ == '__main__':
    opt.model.flag = 'attention'
    opt.model.model = "cls_so3net_pn"

    if opt.mode == 'train':
        # overriding training parameters here
        opt.batch_size = 8 #12
        opt.train_lr.decay_rate = 0.5
        opt.train_lr.decay_step = 20000
        opt.train_loss.attention_loss_type = 'default'

    trainer = Trainer(opt)

    # optional, wandb
    wandb.init(
        project='train_modelnet_cls',
        config={
        "architecture": 'cls_so3net_pn',
        "dataset": 'modelnet',})

    if opt.mode == 'train':
        trainer.train()
    elif opt.mode == 'eval':
        trainer.eval() 
    

    # optional, wandb
    wandb.finish()

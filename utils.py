import wandb as online_logger

def exp_decay(epoch):
    k = 0.999998
    return k**(epoch)

online_logger.init('dqn-learning')

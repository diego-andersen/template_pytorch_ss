from .standard_trainer import StandardTrainer

def create_trainer(opt):
    return StandardTrainer(opt)
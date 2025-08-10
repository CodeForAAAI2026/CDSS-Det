import torch.nn as nn

class EnsembleTSModel(nn.Module):
    def __init__(self, model, teacher_model, pixpro_model, discriminator):
        super(EnsembleTSModel, self).__init__()

        self.model = model
        self.teacher_model = teacher_model
        self.pixpro_model = pixpro_model
        self.discriminator = discriminator 
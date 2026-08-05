import torch
from torch import nn

from toy_diffusion.models.unet import UNet


class ClassConditionedUNet(UNet):
    def __init__(
        self,
        num_classes,
        class_embedding_dim=32,
        classifier_free_dropout=0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._num_classes = num_classes
        # + 1 class for Classifier-Free Guidance
        self._class_embedding = nn.Embedding(num_classes + 1, class_embedding_dim)
        self._class_proj = nn.Sequential(
            nn.Linear(class_embedding_dim, self.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim * 4, self.time_embedding_dim),
        )
        self._cf_dropout = classifier_free_dropout

    def forward(self, x, timestep, class_label):

        if self.training:
            null_class = self._num_classes
            mask = (
                torch.rand(class_label.shape, device=class_label.device)
                > self._cf_dropout
            )
            class_label = torch.where(
                mask, class_label, torch.ones_like(class_label) * null_class
            )

        class_labels_embed = self._class_embedding(class_label)
        class_labels_embed = self._class_proj(class_labels_embed)

        time_embed = self.get_time_embedding(timestep)

        embedding = time_embed + class_labels_embed

        return self.forward_with_embedding(x, embedding)

# https://github.com/facebookresearch/dinov2/tree/main
# https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
# https://github.com/openai/CLIP

import torch
from torch import nn
import clip


# ------------------------------------------ Backbone: DINOv2 S ------------------------------------------

class Model_DINOv2S_1_0(nn.Module):
    # The one model that (at least) would be tested exactly as it is for every approach
    def __init__(self):
        super().__init__()
        
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') # DINOv2 Small
        embedding_dimension = 384

        self.feature_extractor = dinov2_vits14
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(embedding_dimension, 128)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.leakyReLU(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class Model_DINOv2S_1_LN(nn.Module):
    def __init__(self):
        super().__init__()
        
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') # DINOv2 Small
        embedding_dimension = 384

        self.feature_extractor = dinov2_vits14
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(embedding_dimension, 128)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0.0)

        self.ln1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.leakyReLU(self.ln1(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x


class Model_DINOv2S_1_LNdrop(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') # DINOv2 Small
        embedding_dimension = 384

        self.feature_extractor = dinov2_vits14
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(embedding_dimension, 128)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0.0)

        self.ln1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.leakyReLU(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.sigmoid(self.fc2(x))
        return x


class Model_DINOv2S_3_0(nn.Module):
    def __init__(self):
        super().__init__()

        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') # DINOv2 Small
        embedding_dimension = 384

        self.feature_extractor = dinov2_vits14
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(embedding_dimension, 128)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = nn.Linear(128, 128)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc2.bias, 0.0)

        self.fc3 = nn.Linear(128, 128)
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc3.bias, 0.0)

        self.fc4 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.leakyReLU(self.fc1(x))
        x = self.leakyReLU(self.fc2(x))
        x = self.leakyReLU(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


class Model_DINOv2S_3_LN(nn.Module):
    def __init__(self):
        super().__init__()

        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') # DINOv2 Small
        embedding_dimension = 384

        self.feature_extractor = dinov2_vits14
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(embedding_dimension, 128)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0.0)

        self.ln1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 128)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc2.bias, 0.0)

        self.ln2 = nn.LayerNorm(128)

        self.fc3 = nn.Linear(128, 128)
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc3.bias, 0.0)

        self.ln3 = nn.LayerNorm(128)

        self.fc4 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.leakyReLU(self.ln1(self.fc1(x)))
        x = self.leakyReLU(self.ln2(self.fc2(x)))
        x = self.leakyReLU(self.ln3(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x


class Model_DINOv2S_3_LNdrop(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()

        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') # DINOv2 Small
        embedding_dimension = 384

        self.feature_extractor = dinov2_vits14
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(embedding_dimension, 128)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0.0)

        self.ln1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(128, 128)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc2.bias, 0.0)

        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.fc3 = nn.Linear(128, 128)
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc3.bias, 0.0)

        self.ln3 = nn.LayerNorm(128)
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.fc4 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.leakyReLU(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.leakyReLU(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.leakyReLU(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.sigmoid(self.fc4(x))
        return x


# ------------------------------------------ Backbone: DINOv2 L ------------------------------------------

class Model_DINOv2L_1_0(nn.Module):
    def __init__(self):
        super().__init__()
        
        dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        embedding_dimension = 1024

        self.feature_extractor = dinov2_vitl14
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(embedding_dimension, 128)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.leakyReLU(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class Model_DINOv2L_1_LN(nn.Module):
    def __init__(self):
        super().__init__()
        
        dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        embedding_dimension = 1024

        self.feature_extractor = dinov2_vitl14
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(embedding_dimension, 128)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0.0)

        self.ln1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.leakyReLU(self.ln1(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x


# ------------------------------------------ Backbone: CLIP ViT-L/14 ------------------------------------------

class Model_CLIPvitL_1_0(nn.Module):
    def __init__(self, 
                 pretrained_clip_model: str = "ViT-L/14", 
                 freeze_clip: bool = True,
                 hidden_dim: int = 128):
        super().__init__()
        
        # Load CLIP model
        self.clip_model, _ = clip.load(pretrained_clip_model, device="cpu")
        
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # classifier head
        self.fc1 = nn.Linear(self.clip_model.visual.output_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper feature normalization"""
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        x = self.fc1(image_features.float())
        x = self.leakyReLU(x)
        return self.sigmoid(self.fc2(x))

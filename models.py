############################
import torch
from vit_pytorch import ViT
from vit_pytorch.efficient import ViT
from x_transformers import Encoder
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.cait import CaiT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.cross_vit import CrossViT
from vit_pytorch.pit import PiT
from vit_pytorch.nest import NesT
from vit_pytorch.vit import ViT
from vit_pytorch.recorder import Recorder
from vit_pytorch.efficient import ViT
from nystrom_attention import Nystromformer

#img = torch.randn(1, 3, 32, 32)


def getModel(val = 0):
    if val == 0:
        import torch
        from vit_pytorch import ViT

        v = ViT(
            image_size = 32,
            patch_size = 8,
            num_classes = 2000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        # img = torch.randn(1, 3, 32, 32)
        # preds = v(img) # (1, 1000)
        # print('preds',preds.shape)
        ################################
    elif val == 1:
        v = DeepViT(
            image_size = 32,
            patch_size = 8,
            num_classes = 2000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        #img = torch.randn(1, 3, 32, 32)
        #preds = v(img) # (1, 1000)
        #print('preds',preds.shape)
        ###################################
    elif val == 2:
        v = CaiT(
            image_size = 256,
            patch_size = 16,
            num_classes = 5000,
            dim = 2048,
            depth = 12,             # depth of transformer for patch to patch attention only
            cls_depth = 2,          # depth of cross attention of CLS tokens to patch
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05    # randomly dropout 5% of the layers
        )
        #img = torch.randn(1, 3, 32, 32)
        #preds = v(img) # (1, 1000)
        #print('preds',preds.shape)
        ################################
    elif val == 3:
        v = T2TViT(
            dim = 512,
            image_size = 32,
            depth = 8,
            heads = 16,
            mlp_dim = 512,
            num_classes = 2000,
            t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
        )
        #img = torch.randn(1, 3, 32, 32)
        #preds = v(img) # (1, 1000)
        #print('preds',preds.shape)
        #########################################
    elif val == 4:
        v = CrossViT(
            image_size = 32,
            num_classes = 2000,
            depth = 4,               # number of multi-scale encoding blocks
            sm_dim = 192,            # high res dimension
            sm_patch_size = 8,      # high res patch size (should be smaller than lg_patch_size)
            sm_enc_depth = 2,        # high res depth
            sm_enc_heads = 8,        # high res heads
            sm_enc_mlp_dim = 2048,   # high res feedforward dimension
            lg_dim = 384,            # low res dimension
            lg_patch_size = 8,      # low res patch size
            lg_enc_depth = 3,        # low res depth
            lg_enc_heads = 8,        # low res heads
            lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
            cross_attn_depth = 2,    # cross attention rounds
            cross_attn_heads = 8,    # cross attention heads
            dropout = 0.1,
            emb_dropout = 0.1
        )
        #img = torch.randn(1, 3, 32, 32)
        #pred = v(img) # (1, 1000)
        #print('preds',preds.shape)
        #################################
    elif val == 5:
        v = PiT(
            image_size = 32,
            patch_size = 8,
            dim = 256,
            num_classes = 2000,
            depth = (3, 3, 3),     # list of depths, indicating the number of rounds of each stage before a downsample
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        # forward pass now returns predictions and the attention maps
        #img = torch.randn(1, 3, 32, 32)
        #preds = v(img) # (1, 1000)
        #print('preds',preds.shape)
        ##############################################################
    elif val == 6:
        v = NesT(
            image_size = 32,
            patch_size = 8,
            dim = 96,
            heads = 9,
            num_hierarchies = 3,        # number of hierarchies
            block_repeats = (8, 4, 1),  # the number of transformer blocks at each heirarchy, starting from the bottom
            num_classes = 2000
        )

        #img = torch.randn(1, 3, 32, 32)

        #pred = nest(img) # (1, 1000)
        #print('preds',preds.shape)
        ################################
    elif val == 7:
        from vit_pytorch.vit import ViT
        v = ViT(
            image_size = 32,
            patch_size = 8,
            num_classes = 2000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        # import Recorder and wrap the ViT
        from vit_pytorch.recorder import Recorder

        v = Recorder(v)
        
        # forward pass now returns predictions and the attention maps
        #img = torch.randn(1, 3, 32, 32)
        #preds, attns = v(img)
        #print('preds',preds.shape)
        # there is one extra patch due to the CLS token
        #print('attns',attns.shape) # (1, 6, 16, 65, 65) - (batch x layers x heads x patch x patch)
        ################################################
    elif val == 8:
        from vit_pytorch.efficient import ViT
        from nystrom_attention import Nystromformer

        efficient_transformer = Nystromformer(
            dim = 512,
            depth = 12,
            heads = 16,
            num_landmarks = 256
        )

        v = ViT(
            dim = 512,
            image_size = 32,
            patch_size = 8,
            num_classes = 2000,
            transformer = efficient_transformer
        )

        #img = torch.randn(1, 3, 32, 32) # your high resolution picture
        #preds = v(img) # (1, 1000)
        #print('preds',preds.shape)
        #####################
    elif val == 9:
        from vit_pytorch.efficient import ViT
        from x_transformers import Encoder
        v = ViT(
            dim = 512,
            image_size = 32,
            patch_size = 8,
            num_classes = 2000,
            transformer = Encoder(
                dim = 512,                  # set to be the same as the wrapper
                depth = 12,
                heads = 16,
                ff_glu = True,              # ex. feed forward GLU variant https://arxiv.org/abs/2002.05202
                residual_attn = True        # ex. residual attention https://arxiv.org/abs/2012.11747
            )
        )

        #img = torch.randn(1, 3, 32, 32)
        #preds = v(img) # (1, 1000)
        #print('preds',preds.shape)
    
    
    return v


def predict(model, img):
    out = model(img)
    if type(out) == type((1,2)):
        return out[0]
    return out

def iterateModels():
    for i in range(10):
        img = torch.randn(1, 3, 32, 32)
        model = getModel(i)
        print(i,predict(model,img).shape)



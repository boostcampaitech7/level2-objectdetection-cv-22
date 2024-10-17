import torch
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone import ShapeSpec
from swin_transformer import SwinTransformer  # Swin Transformer 코드를 별도 파일에 저장하고 불러옴

@BACKBONE_REGISTRY.register()
class D2SwinTransformer(Backbone):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()

        # Swin Transformer를 생성하는 부분 (원하는 하이퍼파라미터로 설정 가능)
        self.swin_transformer = SwinTransformer(
            pretrain_img_size=cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE,
            patch_size=cfg.MODEL.SWIN.PATCH_SIZE,
            in_chans=cfg.MODEL.SWIN.IN_CHANS,
            embed_dim=cfg.MODEL.SWIN.EMBED_DIM,
            depths=cfg.MODEL.SWIN.DEPTHS,
            num_heads=cfg.MODEL.SWIN.NUM_HEADS,
            window_size=cfg.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=cfg.MODEL.SWIN.MLP_RATIO,
            qkv_bias=cfg.MODEL.SWIN.QKV_BIAS,
            drop_rate=cfg.MODEL.SWIN.DROP_RATE,
            attn_drop_rate=cfg.MODEL.SWIN.ATTN_DROP_RATE,
            drop_path_rate=cfg.MODEL.SWIN.DROP_PATH_RATE,
            norm_layer=torch.nn.LayerNorm,
            ape=cfg.MODEL.SWIN.APE,
            patch_norm=cfg.MODEL.SWIN.PATCH_NORM,
            out_indices=cfg.MODEL.SWIN.OUT_INDICES,
            frozen_stages=cfg.MODEL.SWIN.FROZEN_STAGES,
            use_checkpoint=cfg.MODEL.SWIN.USE_CHECKPOINT
        )

        # Swin Transformer의 출력 정보를 정의 (각 스테이지의 채널과 스트라이드)
        self._out_feature_strides = {"p0": 4, "p1": 8, "p2": 16, "p3": 32}
        self._out_feature_channels = {
            "p0": self.swin_transformer.num_features[0],
            "p1": self.swin_transformer.num_features[1],
            "p2": self.swin_transformer.num_features[2],
            "p3": self.swin_transformer.num_features[3]
        }
    
    def forward(self, x):
        # Swin Transformer의 forward 호출
        return self.swin_transformer(x)
    
    # Detectron2의 output shape를 정의하는 함수
    def output_shape(self):
        return {
            "p0": ShapeSpec(channels=self._out_feature_channels["p0"], stride=self._out_feature_strides["p0"]),
            "p1": ShapeSpec(channels=self._out_feature_channels["p1"], stride=self._out_feature_strides["p1"]),
            "p2": ShapeSpec(channels=self._out_feature_channels["p2"], stride=self._out_feature_strides["p2"]),
            "p3": ShapeSpec(channels=self._out_feature_channels["p3"], stride=self._out_feature_strides["p3"]),
        }
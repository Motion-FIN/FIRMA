from .attention_encoder import TransformerEncoder

def make_attention_encoder(cfg):
    return TransformerEncoder(cfg)
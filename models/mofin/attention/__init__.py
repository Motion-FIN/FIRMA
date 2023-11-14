from .self_attention import SelfAttention

def make_attention(cfg):
    return SelfAttention(cfg)
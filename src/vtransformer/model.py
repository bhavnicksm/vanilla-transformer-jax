import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax, random, partial

import flax 
import flax.linen as nn

from .layers import *

class Transformer(nn.Module):
  embed_dim : int = 512
  output_dim : int = 512
  src_vocab_size : int = 10000
  trg_vocab_size : int = 10000
  max_length : int = 512 
  n_enc_layers : int = 6
  n_dec_layers : int = 6
  d_k : int = 64
  d_v : int = 64
  d_ff : int = 2048

  

  def setup(self):
    self.word_emb_src = nn.Embed(self.src_vocab_size, self.embed_dim)
    self.word_emb_trg = nn.Embed(self.trg_vocab_size, self.embed_dim)
    self.pos_emb = nn.Embed(self.max_length, self.embed_dim)
    self.encoder_layers = [EncoderBlock(name=f"EncoderBlock_{i}") for i in range(self.n_enc_layers)]
    self.decoder_layers = [DecoderBlock(name=f"DecoderBlock_{i}") for i in range(self.n_dec_layers)]
    self.fc_out = nn.Dense(self.output_dim)

  def __call__(self, src_input, trg_input, mask):
    enc_out = self.encoder(src_input)
    dec_out = self.decoder(trg_input, enc_out, mask)
    return dec_out
  
  def encoder(self, input):
    pos = jnp.arange(input.shape[-1]).reshape(1,-1).repeat(input.shape[0], axis=0)
    pos_enc = self.pos_emb(pos)
    word_enc = self.word_emb_src(input)

    x = word_enc + pos_enc
    for i in range(len(self.encoder_layers)):
      x = self.encoder_layers[i](x)
    return x
  
  def decoder(self, input, encoding, mask):
    pos = jnp.arange(input.shape[-1]).reshape(1,-1).repeat(input.shape[0], axis=0)
    pos_enc = self.pos_emb(pos)
    word_enc = self.word_emb_trg(input)

    x = word_enc + pos_enc
    for i in range(len(self.decoder_layers)):
      x = self.decoder_layers[i](x, encoding, mask)
    x = self.fc_out(x)
    return x
    
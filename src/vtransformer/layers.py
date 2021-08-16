import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax, random, partial

import flax 
import flax.linen as nn


class MultiHeadSelfAttention(nn.Module):
  n_heads : int = 8
  d_k : int = 64
  d_v : int = 64
  output_dim : int = 512 

  @nn.compact  
  def __call__ (self, input: jnp.ndarray):
    heads = []

    for i in range(self.n_heads):
      keys_i = nn.Dense(self.d_k, name=f"Head_{i}_keys")(input)
      query_i = nn.Dense(self.d_k, name=f"Head_{i}_query")(input)
      values_i = nn.Dense(self.d_v, name=f"Head_{i}_value")(input)
      x = self.ScaledDotProductAttention(keys_i, query_i, values_i)
      heads.append(x)
      
    heads = jnp.concatenate(heads, axis=-1)
    output = nn.Dense(self.output_dim, name="output_layer")(heads)    
    return output

  def ScaledDotProductAttention(self, keys : jnp.ndarray, query : jnp.ndarray, values : jnp.ndarray):
    x = jnp.einsum("...ik, ...jk -> ...ij", query, keys) /jnp.sqrt(self.d_k)
    x = jnp.matmul(nn.softmax(x), values)
    return x   

class MaskedMultiHeadSelfAttention(nn.Module):
  n_heads : int = 8
  d_k : int = 64
  d_v : int = 64
  output_dim : int = 512 

  @nn.compact  
  def __call__ (self, input: jnp.ndarray, mask : jnp.ndarray):
    heads = []

    for i in range(self.n_heads):
      keys_i = nn.Dense(self.d_k, name=f"Head_{i}_keys")(input)
      query_i = nn.Dense(self.d_k, name=f"Head_{i}_query")(input)
      values_i = nn.Dense(self.d_v, name=f"Head_{i}_value")(input)
      x = self.MaskedScaledDotProductAttention(keys_i, query_i, values_i, mask)
      heads.append(x)
      
    heads = jnp.concatenate(heads, axis=-1)
    output = nn.Dense(self.output_dim, name="output_layer")(heads)    
    return output

  def MaskedScaledDotProductAttention(self, keys : jnp.ndarray, query : jnp.ndarray, values : jnp.ndarray, mask : jnp.ndarray):
    x = jnp.einsum("...ik, ...jk -> ...ij", query, keys) /jnp.sqrt(self.d_k)
    x = nn.softmax(x)
    x = jnp.einsum("...jk, j -> ...jk", x, mask)
    x = jnp.matmul(x, values)
    return x   

class MultiHeadAttention(nn.Module):
  n_heads : int = 8
  d_k : int = 64
  d_v : int = 64
  output_dim : int = 512 

  @nn.compact  
  def __call__ (self, keys : jnp.ndarray, query : jnp.ndarray, values : jnp.ndarray):
    heads = []

    for i in range(self.n_heads):
      keys_i = nn.Dense(self.d_k, name=f"Head_{i}_keys")(keys)
      query_i = nn.Dense(self.d_k, name=f"Head_{i}_query")(query)
      values_i = nn.Dense(self.d_v, name=f"Head_{i}_value")(values)
      x = self.ScaledDotProductAttention(keys_i, query_i, values_i)
      heads.append(x)
      
    heads = jnp.concatenate(heads, axis=-1)
    output = nn.Dense(self.output_dim, name="output_layer")(heads)    
    return output

  def ScaledDotProductAttention(self, keys : jnp.ndarray, query : jnp.ndarray, values : jnp.ndarray):
    x = jnp.einsum("...ik, ...jk -> ...ij", query, keys) /jnp.sqrt(self.d_k)
    x = jnp.matmul(nn.softmax(x), values)
    return x   


class PositionwiseFeedForward(nn.Module):
  hidden_dim : int = 2048
  output_dim : int = 512

  @nn.compact
  def __call__(self, input):
    x = input
    x = nn.Dense(self.hidden_dim)(x)
    x = nn.relu(x)
    x = nn.Dense(self.output_dim)(x)
    return x

class EncoderBlock(nn.Module):
  emb_dim : int = 512
  n_heads : int = 8
  d_k     : int = 64
  d_v     : int = 64
  d_ff    : int = 2048
  

  @nn.compact
  def __call__(self, input):
    x = input
    x = MultiHeadSelfAttention(self.n_heads, self.d_k, self.d_v, self.emb_dim)(x)
    x = x + input
    x = nn.LayerNorm()(x)

    y = x.copy()
    y = PositionwiseFeedForward(self.d_ff, self.emb_dim)(y)
    y = y + x
    y = nn.LayerNorm()(y)

    return y


class DecoderBlock(nn.Module):
  emb_dim : int = 512
  d_k : int = 64
  d_v : int = 64
  d_ff : int = 2048
  n_heads : int = 8  

  @nn.compact
  def __call__(self, input : jnp.ndarray, encoding : jnp.ndarray,  mask : jnp.ndarray):
    x = input
    x = MaskedMultiHeadSelfAttention(n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v, output_dim=self.emb_dim)(x, mask)
    x = x + input
    x = nn.LayerNorm()(x)

    y = x.copy()
    y = MultiHeadAttention(n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v, output_dim=self.emb_dim)(encoding, y, encoding)
    y = y + x
    y = nn.LayerNorm()(y)

    z = y.copy()
    z = PositionwiseFeedForward(hidden_dim=self.d_ff, output_dim=self.emb_dim)(z)
    z = z + y
    z = nn.LayerNorm()(z)

    return z


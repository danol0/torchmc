# torchmc

Implementing [ChEES-HMC](http://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf) in PyTorch.

Because:

> NUTS’s relatively complex control flow makes it difficult to efficiently run many parallel chains on accelerators such as GPUs.

> Implementing parallel-chain HMC on top of a vector-oriented library such as TensorFlow, JAX, or PyTorch that supports automatic differentiation is relatively easy as long as the chains run in lockstep; a batch of gradients for the chains can be computed in parallel quickly on SIMD accelerators such as GPUs or TPUs (Lao et al., 2020).
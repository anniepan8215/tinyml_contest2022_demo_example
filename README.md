# Python example code on MACPRO M1

## What's in this repository/branch?

Macbook Pro M1 update the CUDA with MPS

When you use torch and send model to the device, remember doing follow instead:

    device = torch.device("mps")

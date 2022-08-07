# Python example code on MACPRO M1

## CUDA in MAC

Macbook Pro M1 update the CUDA with MPS

When you use torch and send model to the device, remember doing follow instead:

    device = torch.device("mps")

## create related file

In original file, when we train the data, we need to create file named "saved_models" to save the related mode.

In this mac_file, code added following to create "saved_models" automatecally

    if not os.path.isdir('./saved_models'):
        os.mkdir('./saved_models')

## Code in testing_performances.py

Change code from

    net.cuda()
        device = torch.device('cuda:0')

To

    net = net.float().to(device)
       

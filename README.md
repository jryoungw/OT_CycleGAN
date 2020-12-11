# OT_CycleGAN (Optimal Transport-driven CycleGAN)

Optimal Transport-driven CycleGAN [[Paper](https://arxiv.org/abs/1909.12116)]

## Train on custom dataset

<pre><code>
python train.py --data_path directory_for_custom_dataset --domain_A subdirctory_for_domain_A \\
                --domain_B subdirectory_for_domain_B --extension png
</code></pre>

# Details about implementation

1. **data_path** should contain **domain_A** and **domain_B** as its subdirectories.
2. **extension** should be able to open with **numpy.load** or **SimpleITK.ReadImage**. If your file extension is not supported by these functions, modify [[utils.py](https://github.com/jryoungw/OT_CycleGAN/blob/d0bbb2cc6481bf76b10eb720826c2eb6ab7b65d7/utils.py#L20)].
3. **noramlize** only supports one of **minmax, tanh, CT, None**. If you want to normalize with different method, modify [[utils.py](https://github.com/jryoungw/OT_CycleGAN/blob/d0bbb2cc6481bf76b10eb720826c2eb6ab7b65d7/utils.py#L20)]. CT stands for computed tomography in medical imaging.
4. This repository is only implemented with **Figure 7** and **Table 1(d)** of the original [Paper](https://arxiv.org/abs/1909.12116). Especially, I only implemented discriminator loss for **Equation 3.24**.
5. With my personal experience, I changed **Algorithm 3.1** to train the generator prior to the discriminator. Training generator prior to discriminator makes model more stable.

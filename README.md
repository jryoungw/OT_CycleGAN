# OT-CycleGAN (Optimal Transport-driven CycleGAN)

Pytorch implementation of **Optimal Transport driven CycleGAN for Unsupervised Learning in Inverse Problems** [[Paper](https://arxiv.org/abs/1909.12116)]

# Train on custom dataset

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
6. If you train model with **CT** option for **normalize** configuration, you may want to convert numpy ndarray into DICOM format. It might be helpful referencing [this code](https://github.com/jryoungw/DicomWrite/blob/master/write_dicom.py)
7. Alert! *This is not official implementation*. 


# Training Result

I only experimented CT image for model training. However, for legal issue - such as privacy of the patient, I cannot upload converted CT image.

In further commits, I would be happy to share other images that have no legal issue.

# Changes

2021.01.07 : Added skip connection in [ConvBlock](https://github.com/jryoungw/OT_CycleGAN/blob/master/models.py#L5)

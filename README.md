# OT_CycleGAN

Optimal Transport-driven CycleGAN [[Paper](https://arxiv.org/abs/1909.12116)]

## Train on custom dataset

<pre><code>
python train.py --data_path directory_for_custom_dataset --domain_A subdirctory_for_domain_A \\
                --domain_B subdirectory_for_domain_B --extension png
</code></pre>

1. **data_path** should contain **domain_A** and **domain_B** as its subdirectory.
2. **extension** should be able to open with **numpy.load** or **SimpleITK.ReadImage**. If your file extension is not supported by these functions, modify [[utils.py](https://github.com/jryoungw/OT_CycleGAN/blob/d0bbb2cc6481bf76b10eb720826c2eb6ab7b65d7/utils.py#L20)]

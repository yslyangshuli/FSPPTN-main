# FSPPTN [[Paper Link]] 

Shuli Yang, Shu Tang*, Xinbo Gao, Xianzhong Xie, Jiaxu Leng

This repository is an official implementation of the paper "A Lightweight Frequency-Selection-based Progressive Patch Transformer Network for Single Image Super-Resolution".


# News: FSPPTN is now available!

## Environment
- Platforms: Ubuntu, CUDA >= 11.8
- python >= 3.10
- PyTorch >= 2.1
- BasicSR == 1.4.2

### Installation
```bash
pip install -r requirements.txt
python setup.py develop
```
## How To Train

- Refer to `./options/train/FSPPTN` for the configuration file of the model to train.
- The single GPU training command is as follows:

```
python basicsr/train.py -opt options/train/FSPPTN/train_FSPPTN_SRx2_scratch.yml
```
```
python basicsr/train.py -opt options/train/FSPPTN/train_FSPPTN_SRx3_scratch.yml
```
```
python basicsr/train.py -opt options/train/FSPPTN/train_FSPPTN_SRx4_scratch.yml
```

- The distributed training command is as follows:

```
python -m torch.distributed.run --nproc_per_node=2 --master_port=18584 basicsr/train.py -opt options/train/FSPPTN/train_FSPPTN_SRx2_scratch.yml --launcher pytorch --auto_resume
```
```
python -m torch.distributed.run --nproc_per_node=2 --master_port=18584 basicsr/train.py -opt options/train/FSPPTN/train_FSPPTN_SRx3_scratch.yml --launcher pytorch --auto_resume
```
```
python -m torch.distributed.run --nproc_per_node=2 --master_port=18584 basicsr/train.py -opt options/train/FSPPTN/train_FSPPTN_SRx4_scratch.yml --launcher pytorch --auto_resume
```

The training logs and weights will be saved in the `./experiments` folder.

## How To Test 

- Refer to `./options/test/FSPPTN` for the configuration file of the model to be tested.

```
python basicsr/test.py -opt options/test/test_FSPPTN_x2.yml
```
```
python basicsr/test.py -opt options/test/test_FSPPTN_x3.yml
```
```
python basicsr/test.py -opt options/test/test_FSPPTN_x4.yml
```

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```

```

## Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR). 

Thanks for their awesome works.

## Contact

If you have any question, please email tangshu@cqupt.edu.cn.

# FSPPTN-main
Installation 
pip install -r requirements.txt
python setup.py develop

Training
# x2 scratch
python -m torch.distributed.run --nproc_per_node=2 --master_port=18584 basicsr/train.py -opt options/train/FSPPTN/train_FSPPTN_SRx2_scratch.yml --launcher pytorch --auto_resume

# x3 scratch
python -m torch.distributed.run --nproc_per_node=2 --master_port=18584 basicsr/train.py -opt options/train/FSPPTN/train_FSPPTN_SRx3_scratch.yml --launcher pytorch --auto_resume

# x4 scratch
python -m torch.distributed.run --nproc_per_node=2 --master_port=18584 basicsr/train.py -opt options/train/FSPPTN/train_FSPPTN_SRx4_scratch.yml --launcher pytorch --auto_resume

Testing 
python basicsr/test.py -opt options/test/test_FSPPTN_x2.yml
python basicsr/test.py -opt options/test/test_FSPPTN_x3.yml
python basicsr/test.py -opt options/test/test_FSPPTN_x4.yml

Citation
Please cite us if our work is useful for your research.


Acknowledgements
This code is built in BasicSR.

# MultiTrans
This repository includes the official project of our paper: MultiTrans: Multi-Branch Transformer Decoder Network for Medical Image Segmentation.

## Usage

### 1. Download pre-trained Resnet models

Please note that when loading pre-trained Resnet models and the preprocessed dataset, we use absolute paths in our code. So you can put the pre-trained models and the dataset under any path. Then, please modify the file path of them in the code, and their locations in the code can be easily found according to the error message.

resnet50-deep-stem:[link](https://drive.google.com/file/d/1OktRGqZ15dIyB2YTySLfOVtprerHgbef/view?usp=sharing)

resnet50:[link](https://drive.google.com/file/d/1fUAuRfewRpaS5mFX_IQqrE2syEn9PXrv/view?usp=sharing)

resnet34:[link](https://drive.google.com/file/d/18Erx_ISMt1XMjJlgl4SQsr-iMvcN-7bZ/view?usp=sharing)

resnet18-deep-stem:[link](https://drive.google.com/file/d/1q1VBV37acIte0GynoS054BWfwwdx1NiZ/view?usp=sharing)

resnet18:[link](https://drive.google.com/file/d/1LCybGjJ_d-nALvciBBkZil_XfO-7ptAE/view?usp=sharing)

### 2. Prepare data

- Download the dataset from [official website](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789). Convert them to numpy format, clip within [-125, 275], normalize each 3D volume to [0, 1], and extract 2D slices from 3D volume for training while keeping the testing 3D volume in h5 format.

- Directly use [preprocessed data](https://drive.google.com/file/d/1XjHzJageFKFN7Tg-6F2NJz2sj9hSLPK0/view?usp=sharing) provided by [TransUNet](https://github.com/Beckschen/TransUNet).

### 3. Environment

We trained our model on one NVIDIA GeForce GTX 3090 with the CUDA 11.1 and CUDNN 8.0.

Python 3.8.13.

PyTorch 1.8.1. 

Please refer to 'requirements.txt' for other dependencies.

### 4. Test our trained model 

Download the trained model:[link](https://drive.google.com/drive/folders/17Zs8F6pKSt5C6BAo0uPhsCssobsiE98S?usp=sharing). Put 'epoch_149.pth' into this file: 'Results/model_Trained/MultiTrans_Synapse224/Model/MultiTrans_pretrain_resnet50_deep_V10_epo150_bs24_224_s12100'. Run the following order.

```bash
cd MultiTrans
```

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --Model_Name MultiTrans --backbone resnet50_deep --branch_in_channels 256 512 512 1024 --branch_out_channels 256 --branch_key_channels 16 32 64 128 --seed 12100
```

### 5. Train/Test by yourself

```bash
cd MultiTrans
```

- Run the train script.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --Model_Name MultiTrans --backbone resnet50_deep --branch_in_channels 256 512 512 1024 --branch_out_channels 256 --branch_key_channels 16 32 64 128 --seed 1290
```

- Run the test script.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --Model_Name MultiTrans --backbone resnet50_deep --branch_in_channels 256 512 512 1024 --branch_out_channels 256 --branch_key_channels 16 32 64 128 --seed 1290
```

### 5. Ablation Experiments of Table 2,3,4,5 in our paper

Add following orders to train-script and test-script.

- Use a single Transformer branch:

```bash
--branch_choose 0   # 0 or 1 or 2 or 3
```

- Remove one of the four branches:

```bash
--branch_choose 0 1 3   # 1 2 3 or 0 2 3 or 0 1 3 or 0 1 2
```

- Ablation experiments on the design of efficient self-attention. If_efficient_attention: use Order-Changing or not; one_kv_head: use Head-Sharing or not; share_kv: use Projection-Sharing or not:

```bash
--If_efficient_attention True --one_kv_head True --share_kv False   
```

- If you want to replace our efficient self-attention with stand self-attention, you need to train the model on 3 GTX 3090 GPUs with halved batch sizes and base_lr:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --dataset Synapse --Model_Name MultiTrans --backbone resnet50_deep --branch_in_channels 256 512 512 1024 --branch_out_channels 256 --branch_key_channels 16 32 64 128 --If_efficient_attention False --n_gpu 3 --batch_size 4 --base_lr 0.005 --seed 1290
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python test.py --dataset Synapse --Model_Name MultiTrans --backbone resnet50_deep --branch_in_channels 256 512 512 1024 --branch_out_channels 256 --branch_key_channels 16 32 64 128 --If_efficient_attention False --n_gpu 3 --batch_size 4 --base_lr 0.005 --seed 1290
```

## Reference
* [TransUNet](https://github.com/Beckschen/TransUNet)

## Citations

```bibtex

xxx

```

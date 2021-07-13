Sentence Pairs  <img src="https://pytorch.org/assets/images/logo-dark.svg" width = "90" align=center />

###### This project is under working:  Siamese-LSTM | ESIM | BIMPM | MVAN


    
Setup and Dependencies
----------------------

This code is implemented using PyTorch v1.7.0, and provides out of the box support with CUDA 10.2

Anaconda / Miniconda is the recommended to set up this codebase.

### Anaconda or Miniconda

Clone this repository and create an environment:

```shell
git clone https://www.github.com/ecpark4545/SentencePair
conda create -n SentencePair python=3.7

# activate the environment and install all dependencies
conda activate SentencePair
cd SentencePair

# https://pytorch.org
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

Preparing Data & Path
-------------
- Put the data under raw_data dir
- you can go to cfg/base.yaml then adjust save_path or data_path  

Training
-------------
you can simply run the model with this code
```shell
# base model (SiameseLSTM) 
python main.py 
# base model (ESIM) 
python main.py --model esim 
# base model (BIMPM) 
python main.py --model bimpm
# base model (MVAN) 
python main.py --model mvan
```

Tuning
-------------
you can easily tune the hyper parameters with arguments such as batch_size, learning_rate, loss(mse, ce), hidden_size, embedding_size
```shell
# best base model (SiameseLSTM) 
python main.py --bs 64 --lr 1e-3 --hs 128 --es 128 --loss_type mse

# best ESIM 
python main.py --cfg_file cfg/base.yml --bs 32 --gpu 3 --lr 5e-4 --model esim --dr 0.5  --hs 300 --es 300 --loss_type ce --cls

# best BIMPM
python main.py --cfg_file cfg/base.yml --bs 32 --gpu 5 --lr 1e-3 --model bimpm --dr 0.1  --hs 300 --es 100 --loss_type ce --cls

# best MVAN
python main.py --cfg_file cfg/base.yml --bs 128 --gpu 2 --lr 1e-4 --model mvan --dr 0.5  --hs 512 --es 300 --loss_type ce --cls 
```


 Inference
-------------
- before Inference, you should put checkpoint in the right directory and adjust cfg file
- you can use ensemble (voting) method (checkout pred.py)
```shell
# base model (SiameseLSTM) 
python main.py --bs 64 --lr 1e-3 --hs 128 --es 128 --loss_type mse --pred

```
ETC
-------------

- This code borrows heavily from [Siamese-LSTM](https://github.com/MahmoudWahdan/Siamese-Sentence-Similarity), [ESIM](https://github.com/coetaur0/ESIM), [BIMPM](https://github.com/galsang/BIMPM-pytorch), [MVAN](https://github.com/ecpark4545/MVAN-VisDial)  repositories. Many thanks.

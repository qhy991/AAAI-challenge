# AAAI-challenge
The content of training for the contest

For the .onnx files are bigger than the upload limit, I upload the files to Beihang Cloud. [Links](https://bhpan.buaa.edu.cn:443/link/B1A7ECE0C3F03ADDED6FF56F3A7A1897)

The structure of training is https://github.com/Fafa-DL/Awesome-Backbones. It's easier compared with the official code of RepVGG, and there are many others classic models for classification.

Now the training codes are all from the official repo, and I create a new repo [here](https://github.com/qhy991/AAAI) to store the modifications I made for training AAAI dataset.

In addition, show the results below
|Model|Type|Accuracy|Weight|onnx|
|-|-|-|-|-|
|RepVGG|A0|88.23%|-|-|
|RepOpt|B1|90.91%|-|-|
|Repopt|A0-(Hyper-serach-AAAI-dataset)|89.42%|-|-|
|Repopt|A0-(Hyper-serach-CF100-dataset)|89.84%|-|-|


## Train Repopt-A0
For the scale file of A0 isn't released, so I train the scale file. Run the code in terminal below:
```sh
python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 12349 main_repopt.py --data-path /data/AAAI/Awesome-Backbones/datasets --arch RepOpt-VGG-A0-hs --batch-size 32 --tag search --opts TRAIN.EPOCHS 240 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 10 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET imagenet 
```


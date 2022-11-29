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

## Something about the dataset

Through observing the picture number of each class, the long-tailed feature is obvious.
![the_dist_of the_dataset_decreasing](./pic/the_dist_of_the_dataset_decreasing.png)




![the distribution of the dataset](./pic/the_dist_of_the_dataset.png)

**The statistical data:**

|       | num         |
| ----: | ----------- |
| count | 89.000000   |
|  mean | 561.831461  |
|   std | 809.292792  |
|   min | 8.000000    |
|   25% | 135.000000  |
|   50% | 271.000000  |
|   75% | 502.000000  |
|   max | 5121.000000 |

**The resolution of dataset**

Not only is the number of each class unbalanced, but also the resolution of the pictures vary greatly. As the image below shows: 

![the resolution of picture](./pic/The_resolution_of_the_dataset.png)



The channel of the picture is 3. Among all 50003 pictures, there are 11133 different resolution. Most of them are smaller than 200*200.

**The accuracy of each class**

This table show the accuracy of each class with ascending trend. The validation number means the number of each class in validation dataset.

| Class | Accuracy | validation num |
| ----: | -------: | -------------- |
|    81 | 0.125000 | 8              |
|    83 | 0.418803 | 234            |
|    76 | 0.500000 | 2              |
|    63 | 0.558824 | 136            |
|    74 | 0.559322 | 118            |
|    79 | 0.647059 | 17             |
|    86 | 0.666667 | 3              |
|    73 | 0.686441 | 118            |
|    78 | 0.687500 | 16             |
|    62 | 0.699248 | 133            |
|    27 | 0.714286 | 7              |
|    52 | 0.800000 | 10             |
|    64 | 0.815789 | 38             |
|    84 | 0.833333 | 234            |
|    32 | 0.846154 | 13             |
|    82 | 0.850427 | 234            |
|    65 | 0.888889 | 36             |
|    33 | 0.900000 | 10             |
|     8 | 0.904255 | 94             |
|    39 | 0.904762 | 21             |
|    80 | 0.905512 | 127            |
|    41 | 0.906250 | 32             |
|     7 | 0.909091 | 44             |
|    14 | 0.909091 | 22             |
|     5 | 0.909091 | 11             |
|    70 | 0.911765 | 306            |
|    72 | 0.915254 | 118            |
|    53 | 0.920000 | 25             |
|    75 | 0.922780 | 259            |
|    10 | 0.931034 | 58             |
|    24 | 0.935484 | 31             |
|    66 | 0.937500 | 16             |
|    40 | 0.937500 | 16             |
|    85 | 0.939453 | 512            |
|    19 | 0.944444 | 18             |
|    38 | 0.945946 | 37             |
|    20 | 0.958333 | 24             |
|    61 | 0.965517 | 58             |
|    48 | 0.968750 | 32             |
|    18 | 0.975610 | 41             |
|    12 | 0.976190 | 42             |
|    87 | 0.978261 | 46             |
|    50 | 0.979167 | 48             |
|     6 | 0.982906 | 117            |
|    44 | 0.984615 | 65             |
|     4 | 0.987805 | 82             |
|    43 | 0.988372 | 172            |
|    67 | 1.000000 | 29             |
|    68 | 1.000000 | 1              |
|    69 | 1.000000 | 1              |
|    71 | 1.000000 | 115            |
|    60 | 1.000000 | 30             |
|    58 | 1.000000 | 10             |
|    59 | 1.000000 | 210            |
|    77 | 1.000000 | 17             |
|    57 | 1.000000 | 26             |
|     0 | 1.000000 | 22             |
|    55 | 1.000000 | 23             |
|     1 | 1.000000 | 29             |
|     2 | 1.000000 | 37             |
|     3 | 1.000000 | 44             |
|     9 | 1.000000 | 27             |
|    11 | 1.000000 | 32             |
|    13 | 1.000000 | 29             |
|    15 | 1.000000 | 3              |
|    16 | 1.000000 | 18             |
|    17 | 1.000000 | 18             |
|    21 | 1.000000 | 15             |
|    22 | 1.000000 | 19             |
|    23 | 1.000000 | 30             |
|    25 | 1.000000 | 30             |
|    26 | 1.000000 | 7              |
|    28 | 1.000000 | 11             |
|    29 | 1.000000 | 7              |
|    30 | 1.000000 | 25             |
|    31 | 1.000000 | 7              |
|    34 | 1.000000 | 8              |
|    35 | 1.000000 | 7              |
|    36 | 1.000000 | 11             |
|    37 | 1.000000 | 11             |
|    42 | 1.000000 | 25             |
|    45 | 1.000000 | 2              |
|    46 | 1.000000 | 8              |
|    47 | 1.000000 | 45             |
|    49 | 1.000000 | 14             |
|    51 | 1.000000 | 50             |
|    54 | 1.000000 | 60             |
|    56 | 1.000000 | 1              |
|    88 | 1.000000 | 15             |

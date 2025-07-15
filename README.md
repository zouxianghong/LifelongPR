### LifelongPR
* For details, please refer to [Project Main Page](https://zouxianghong.github.io/LifelongPR/)
* Demo video in [Youtube](https://www.youtube.com/watch?v=D7ud0X1kywE)
[![demo](./short-demo.gif)](https://www.youtube.com/watch?v=D7ud0X1kywE)

### Environment
* Docker image:
```
docker pull zouxh22135/pc_loc:v1
```

### Benchmark Datasets
* Seq1: Oxford RobotCar -> DCC -> Riverside -> In-house
* Seq2: Oxford RobotCar -> Hankou -> WHU-Campus -> In-house
* Public datasets: Oxford RobotCar, MulRan (DCC / Riverside), In-house
* Self-collected datasets: Hankou, WHU-Campus, see [PatchAugNet](https://whu-usi3dv.github.io/PatchAugNet/)
* Run python scripts in /generating_queries, such as:
```
python generating_queries/Oxford/generate_train.py --dataset_root [path] --save_folder [path]
python generating_queries/Oxford/generate_test.py --dataset_root [path] --save_folder [path]
```

### Training
* Train on the first dataset:
```
python training/train.py --config config/default.yaml
```
* Train on the remaining datasets:
```
python training/train_incremental.py --config config/default.yaml
```
* Note: 1) parameters, such as model name and memory size, can be modified in the config file

### Testing
* Test on the other dataset:
```
python eval/evaluate.py --config config/default.yaml --ckpt [check point file]
```

### Pre-trained models
* Download in googledrive: [models](https://drive.google.com/drive/folders/1LGbzHPYkFiytN2TgXUrTJK85PjSZn-Z5?usp=sharing)

### Citation
If you find the code or trained models useful, please consider citing:
```
@article{zou2025lifelongpr,
  title={LifelongPR: Lifelong knownledge fusion for point cloud place recognition based on replay and prompt learning},
  author={Zou, Xianghong and Li, Jianping and Chen, Zhe and Cao, Zhen and Dong, Zhen and Qiegen, Liu and Yang, Bisheng},
  journal={Information Fusion},
  volume={xxx},
  pages={xxx--xxx},
  year={2025},
  publisher={Elsevier}
}
```

#### Acknowledgement
Our code is built upon [InCloud](https://github.com/csiro-robotics/InCloud).

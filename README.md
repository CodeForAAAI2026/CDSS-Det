# CDSS-Det
Official implementation of Cross-Domain Semi-Supervised Organ Detection, AAAI 2026 submission


# Installation
```
conda create -n cdss-det python=3.8.18
conda activate cdss-det
conda install pytorch==2.3.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```


# Configs
Config files are under config/aaai, adjust the dataset path according to your storage and other parameters as you needed.

# Datasets
Our preprocessed datasets can be downloaded from [here](https://pan.baidu.com/s/1OlRY_z7R4z2MbcMadYiEXQ).
We don't upload Abdomen-Atlas since it is too large, you can download it from the [official source](https://github.com/MrGiovanni/AbdomenAtlas) instead.

If you use your own dataset, make sure it is preprocessed in the same way as organdetr/data/preprocessor.py.


# Training

## Pre-training
First, pre-train the model using source data.
```
export ORGANDETR_DATA=${your_data_path}
python scripts/sd_train.py --config ${your_config_name}
```

## Cross-Domain training
Second, use the pre-trained model above to train the source, labeled target and unlabeled target data together.
```
export ORGANDETR_DATA=${your_data_path}
python scripts/aaai_ts_cd_train.py --config ${your_config_name}$
``` 

## Inference
```
python scripts/ts_test.py --run ${your_config_name} --model ${your_model_checkpoint_name}
```
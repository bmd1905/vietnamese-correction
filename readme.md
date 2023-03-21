# Vietnamese Correction
[![Inference](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmd1905/vietnamese-correction/blob/main/inference.ipynb?hl=en)
[![Stars](https://img.shields.io/github/stars/bmd1905/Vietnamese-Corrector.svg)](https://api.github.com/repos/bmd1905/vietnamese-correction)

### Error Correction based on [BARTpho](https://github.com/VinAIResearch/BARTpho)

# Overview
## BARTpho
>We present BARTpho with two versions, BARTpho-syllable and BARTpho-word, which are the first public large-scale monolingual sequence-to-sequence models pre-trained for Vietnamese. BARTpho uses the "large" architecture and the pre-training scheme of the sequence-to-sequence denoising autoencoder BART, thus it is especially suitable for generative NLP tasks. We conduct experiments to compare our BARTpho with its competitor mBART on a downstream task of Vietnamese text summarization and show that: in both automatic and human evaluations, BARTpho outperforms the strong baseline mBART and improves the state-of-the-art. We further evaluate and compare BARTpho and mBART on the Vietnamese capitalization and punctuation restoration tasks and also find that BARTpho is more effective than mBART on these two tasks.

For more details, look at the original [paper](https://arxiv.org/abs/2109.09701).

## My Model
This model is a fine-tuned version of ```vinai/bartpho-syllable```. The original dataset is avaiable at [@duyvuleo/VNTC](https://github.com/duyvuleo/VNTC), I customized it for error correction task, you can find my final dataset at [Huggingface Datasets](https://huggingface.co/datasets/bmd1905/error-correction-vi).

# Usage
This model is avaiable in Huggingface at [bmd1905/vietnamese-correction](https://huggingface.co/bmd1905/vietnamese-correction), to quickly use my model, simply run:
```python
from transformers import pipeline

corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")
```
```python
# Example
print(corrector("toi dang là sinh diên nem hay ở truong đạ hoc khoa jọc tự nhiên , trogn năm ke tiep toi sẽ chọn chuyen nganh về trí tue nhana tạo"))
print(corrector("côn viec kin doanh thì rất kho khan nên toi quyết dinh chuyển sang nghê khac  "))
print(corrector("một số chuyen gia tài chính ngâSn hànG của Việt Nam cũng chung quan điểmnày"))
print(corrector("Lần này anh Phươngqyết xếp hàng mua bằng được 1 chiếc"))
print(corrector("Nhưng sức huỷ divt của cơn bão mitch vẫn chưa thấm vào đâu lsovớithảm hoạ tại Bangladesh ăm 1970"))
```
```
Output:
- Tôi đang là sinh viên hay ở trường đại học khoa học tự nhiên, trong năm kế tiếp, tôi sẽ chọn chuyên ngành về trí tuệ nhân tạo.
- Công việc kinh doanh thì rất khó khăn nên tôi quyết định chuyển sang nghê khác.
- Một số chuyên gia tài chính ngân hàng của Việt Nam cũng chung quan điểm này.
- Lần này anh Phương quyết xếp hàng mua bằng được 1 chiếc.
- Nhưng sức huỷ diệt của cơn bão mitch vẫn chưa thấm vào đâu so với thảm hoạ tại Bangladesh năm 1970 .
```
Or you can use my [notebook](https://colab.research.google.com/github/bmd1905/vietnamese-correction/blob/main/inference.ipynb?hl=en).

# Training
First one, you need to install dependencies:
```
pip install -r requirements.txt
```
In case of pretraining on your own custom-dataset, you must modify the format of files the same with [data.vi.txt](https://github.com/bmd1905/vietnamese-correction/blob/main/data/data.vi.txt). You then run the following script to create your dataset:
```
python generate_dataset.py --data path/to/data.txt --language 'vi' --model_name 'vinai/bartpho-syllable'
```
S.t.
* ```data```: path to your formated data.
* ```language```: a string to name your outputed file.
* ```model_name```: check wherever your sentences suitable with the model length, if not, remove it.

When you accomplished creating dataset, let train your model, simply run:
```
python train.py \
      --model_name_or_path bmd1905/vietnamese-correction \
      --do_train \
      --do_eval \
      --evaluation_strategy="steps" \
      --eval_steps=10000 \
      --train_file /data/vi.train.csv \
      --validation_file /data/vi.test.csv \
      --output_dir ./models/my-vietnamese-correction/ \
      --overwrite_output_dir \
      --per_device_train_batch_size=4 \
      --per_device_eval_batch_size=4 \
      --gradient_accumulation_steps=32 \
      --learning_rate="1e-4" \
      --num_train_epochs=2 \
      --predict_with_generate \
      --logging_steps="10" \
      --save_total_limit="2" \
      --max_target_length=1024 \
      --max_source_length=1024 \
      --fp16
```
Alternative way, you can use my Colab notebook [with script](https://colab.research.google.com/github/bmd1905/vietnamese-correction/blob/main/colab_train.ipynb?hl=en) or [with code](https://colab.research.google.com/github/bmd1905/vietnamese-correction/blob/main/aio.ipynb?hl=en).

# References
[1] [BARTpho](https://github.com/VinAIResearch/BARTpho) \
[2] [@oliverguhr/spelling](https://github.com/oliverguhr/spelling) \
[3] [@duyvuleo/VNTC](https://github.com/duyvuleo/VNTC)


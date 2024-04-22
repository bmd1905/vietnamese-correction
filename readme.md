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
MAX_LENGTH = 512

# Define the text samples
texts = [
    "côn viec kin doanh thì rất kho khan nên toi quyết dinh chuyển sang nghề khac  ",
    "toi dang là sinh diên nam hai ở truong đạ hoc khoa jọc tự nhiên , trogn năm ke tiep toi sẽ chọn chuyen nganh về trí tue nhana tạo",
    "Tôi  đang học AI ở trun tam AI viet nam  ",
    "Nhưng sức huỷ divt của cơn bão mitch vẫn chưa thấm vào đâu lsovớithảm hoạ tại Bangladesh ăm 1970 ",
    "Lần này anh Phươngqyết xếp hàng mua bằng được 1 chiếc",
    "một số chuyen gia tài chính ngâSn hànG của Việt Nam cũng chung quan điểmnày",
    "Cac so liệu cho thay ngươi dân viet nam đang sống trong 1 cuôc sóng không duojc nhu mong đọi",
    "Nefn kinh té thé giới đang đúng trươc nguyen co của mọt cuoc suy thoai",
    "Khong phai tất ca nhưng gi chung ta thấy dideu là sụ that",
    "chinh phủ luôn cố găng het suc để naggna cao chat luong nền giáo duc =cua nuoc nhà",
    "nèn kinh te thé giới đang đứng trươc nguy co của mọt cuoc suy thoai",
    "kinh tế viet nam dang dứng truoc 1 thoi ky đổi mơi chưa tung có tienf lệ trong lịch sử"
]

# Batch prediction
predictions = corrector(texts, max_length=MAX_LENGTH)

# Print predictions
for text, pred in zip(texts, predictions):
    print("- " + pred['generated_text'])
```
```
Output:
- Công việc kinh doanh thì rất khó khăn nên tôi quyết định chuyển sang nghề khác.
- Tôi đang là sinh viên hai ở trường đại học khoa học tự nhiên, trong năm kế tiếp, tôi sẽ chọn chuyên ngành về trí tuệ nhân tạo.
- Tôi đang học AI ở trung tâm AI Việt Nam.
- Nhưng sức huỷ diệt của cơn bão mitch vẫn chưa thấm vào đâu so với thảm hoạ tại Bangladesh năm 1970 .
- Lần này anh Phương quyết xếp hàng mua bằng được 1 chiếc.
- Một số chuyên gia tài chính ngân hàng của Việt Nam cũng chung quan điểm này.
- Các số liệu cho thấy ngươi dân Việt Nam đang sống trong 1 cuôc sóng không được nhu mong đọc.
- Niên kinh té thé giới đang đúng trương, nguyên cơ của một cuộc suy thoái.
- Không phai tất ca, nhưng giờ chúng ta thấy điều là sự thật.
- Chính phủ luôn cố găng hết sức để nâng cao chất lượng nền giáo dục của nước nhà.
- Nền kinh tế thế giới đang đứng trước nguy cơ của một cuộc suy thoái.
- Kinh tế Việt Nam đang đứng trước 1 thời kỳ đổi mới, chưa từng có tiền lệ trong lịch sử.
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
Alternative way, you can use [my Colab notebook](https://colab.research.google.com/github/bmd1905/vietnamese-correction/blob/main/train_v2.ipynb?hl=en).


# References
[1] [BARTpho](https://github.com/VinAIResearch/BARTpho) \
[2] [@oliverguhr/spelling](https://github.com/oliverguhr/spelling) \
[3] [@duyvuleo/VNTC](https://github.com/duyvuleo/VNTC)


This repo is sponsored by [AI VIET NAM](https://www.facebook.com/aivietnam.edu.vn).

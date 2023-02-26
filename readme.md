# Vietnamese-Corrector
### Spelling Correction based on [Pretrained BARTpho](https://github.com/VinAIResearch/BARTpho)

# Overview
## BARTpho
>We present BARTpho with two versions, BARTpho-syllable and BARTpho-word, which are the first public large-scale monolingual sequence-to-sequence models >pre-trained for Vietnamese. BARTpho uses the "large" architecture and the pre-training scheme of the sequence-to-sequence denoising autoencoder BART, >thus it is especially suitable for generative NLP tasks. We conduct experiments to compare our BARTpho with its competitor mBART on a downstream task of >Vietnamese text summarization and show that: in both automatic and human evaluations, BARTpho outperforms the strong baseline mBART and improves the >state-of-the-art. We further evaluate and compare BARTpho and mBART on the Vietnamese capitalization and punctuation restoration tasks and also find that >BARTpho is more effective than mBART on these two tasks.

For more details, look at the original [paper](https://arxiv.org/abs/2109.09701).

## My Model


# Usage
First one, you need to install dependencies:
```
pip install -r requirements.txt
```
In case of pretraining on your own custom-dataset, you must modify the format of files the same with [data.vi.txt](https://github.com/bmd1905/Vietnamese-Corrector/blob/main/data/data.vi.txt). You then run the following script to create your dataset:
```
python generate_dataset.py --data path/to/data.txt --language 'vi' --model_name 'vinai/bartpho-word'
```
S.t.
* data: path to your formated data.
* language: a string to name your outputed file.
* model_name: check wherever your sentences suitable with the model length, if not, remove it.

# References
[1] [@oliverguhr/spelling](https://github.com/oliverguhr/spelling) \
[2] [BARTpho](https://github.com/VinAIResearch/BARTpho)


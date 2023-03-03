# JCommonsenseMorality
[日本語 readme](https://github.com/Language-Media-Lab/commonsense-moral-ja/blob/main/README_JP.md)

JCommonsenseMorality is a dataset created through crowdsourcing that reflects the commonsense morality of Japanese annotators.

Dataset is available at [data](https://github.com/Language-Media-Lab/commonsense-moral-ja/tree/main/data).
Code for fine-tuning is available at [finetune.py](https://github.com/Language-Media-Lab/commonsense-moral-ja/tree/main/finetune.py).


Examples
|sentence|label|
|-----|---|
|賽銭箱に石を投げ入れる。(throw a stone in to a offertory box)|1 (wrong)|
|賽銭箱にお賽銭を投げ入れる。(throw money in to a offertory box)|0 (permissible)|
|限定商品を買い占めて転売する。(buy up limited edition items and resell them)|1 (wrong)|
|限定商品を自分の分だけ買う。(buy limited edition items for myself)|0 (permissible)|

## Task and Statistics
All sentences are labeled either '1' or '0', indicating that the described action is clearly morally wrong or permissible, respectively.

Data Statistics
|Train|Dev|Test|Total|
|-----|---|----|-----|
|13,975|1,996|3,992|19,963|

## Baseline
All results are average scores of the model trained on five random seeds.
|Model|acc|pre|rec|f1|
|-----|---|---|---|--|
|[Tohoku BERT base](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking)|0.7836|0.7740|0.7601|0.7664|
|[Tohoku BERT large](https://huggingface.co/cl-tohoku/bert-large-japanese)|0.8033|0.8050|0.7691|0.7860|
|[Waseda RoBERTa large](https://huggingface.co/nlp-waseda/roberta-large-japanese-with-auto-jumanpp)|0.8558|0.8453|0.8481|0.8461|

## License
This work is licensed under a MIT License.
https://github.com/Language-Media-Lab/commonsense-moral-ja/blob/main/LICENSE

## Acknowledgment
This work was supported by JSPS KAKENHI Grant Number JP22J21160.

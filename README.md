# IHS-SP
This is the repository for the "Implicit Hate Speech Detection via Soft Prompt-tuning".

First, by pip install -r requirement.txt to install all the dependencies.

Firstly install OpenPrompt https://github.com/thunlp/OpenPrompt

Then copy prompts/knowledgeable_verbalizer.py to Openprompt/openprompt/prompts/knowledgeable_verbalizer.py

And install BERT model https://huggingface.co/roberta-base to IHS-SP/models/

Also, you can put your own dataset in datasets/TextClassification.

example shell scripts:

python main.py --result_file ./output_fewshot.txt --dataset implicit_hatespeech --template_id 0 --seed 123 --shot 20 --verbalizer union

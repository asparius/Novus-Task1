# Novus-Task1
## FINETUNING
I have used the standart trainer from transformers to finetune the tiny shakespeare dataset. Most of the hyperparameters are default, except the weight decay to prevent overfitting due to small dataset and batch size because of the compute constraints in the google colab. One additional hyparameter is the chunk size. Training dataset is roughly 300k tokens, and splitting it to the size of GPT2 tokenizer is excessive considering it produces less training samples. Therefore I have tried smaller chunk size like 128 or 256 and it works pretty ok. Since dataset is pretty small, I have trained over 10 epochs in which validation loss begins to increase and more epochs exacerbates this. Snippets are in Finetuning.ipynb.

## API and Hosting
I could not host on a cloud platform because my free trial are over almost on all well known platforms. However, at the end of the day hosting locally or on a VM is not so different from each other. Thus, I will be sharing the version that works locally. Finally, model is a outcome of the above finetuning that is present on my huggingface account.

### Creating virtual environment, installing dependencies and launching the API
```
python -m venv task1env
source task1env/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Testing API with curl
Following returns a json that has generated answer.
```
curl -X POST "http://127.0.0.1:8000/generate/" -H "Content-Type: application/json" -d '{"prompt": "Once upon a time", "max_length": 50, "temperature": 0.7}'

```


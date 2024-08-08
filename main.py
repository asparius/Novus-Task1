from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

model_name = "asparius/gpt2-tinyshakespeare"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 256  
    temperature: float = 0.7  

@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    # Tokenize input prompt
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=request.max_length,
            temperature=request.temperature,
            num_return_sequences=1,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}


"""
# How to run 
uvicorn api_for_finetuned_model:app --reload --workers 4

# Doc url
http://127.0.0.1:8000/docs

"""

from fastapi import FastAPI
from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer
from pydantic import BaseModel
from typing import Optional
import re
import string

# CONFIG 
model = AutoModelWithLMHead.from_pretrained('./gpt2-20epoch')
tokenizer = AutoTokenizer.from_pretrained('./gpt2-20epoch')


# Initalizing FastAPI app
app = FastAPI()

# Schema for post request.
class Item(BaseModel):
    text: str
    max_op_words: Optional[int] = 50
    max_time: Optional[float] = 1
    max_predictions: Optional[int] = 1


# Get predictions route
@app.post("/get_predictions/")
async def create_item(item: Item):
    """[summary]

    Args:
        item (Item): [base model schema]

    Returns:
        [type]: [{"generated_text": "Hello How are су"}]
        
    """

    # Conveting input data to dict format
    item_dict = item.dict()

    prompt_text = item_dict['text']
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt")
    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt
        
    output_sequences = model.generate(
        input_ids=input_ids,
        # min_length=15,
        max_length=len(prompt_text) + len(encoded_prompt[0]),
        # temperature=0.3,
        # top_k=0,
        # top_p=0.95,
        # num_beams=5,
        # no_repeat_ngram_size=2,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=3,
        
    )

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    #     # Remove all text after the stop token
    #     text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )
        # total_sequence = total_sequence.split("<EOS>")[0]

        generated_sequences.append(total_sequence)
        print(total_sequence)


    return generated_sequences
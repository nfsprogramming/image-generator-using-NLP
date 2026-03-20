from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import random
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# --- Optional Imports (Heavy NLP) ---
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# --- Setup NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NLP Section: Prompt Generation ---
# --- NLP Section: Prompt Generation ---
prompt_generator = None
model_loaded = False
model_load_error = None

def get_prompt_generator():
    global prompt_generator, model_loaded, model_load_error
    if model_loaded:
        return prompt_generator
    
    if not TRANSFORMERS_AVAILABLE:
        return None

    try:
        print("Loading NLP Model (Lazy Load)...")
        # Full Power: Use GPU if available
        device = 0 if torch.cuda.is_available() else -1
        prompt_generator = pipeline("text-generation", model="distilgpt2", device=device)
        model_loaded = True
        return prompt_generator
    except Exception as e:
        print(f"Warning: NLP model failed to load: {e}")
        model_load_error = str(e)
        model_loaded = True # Stop trying
        return None

def nlp_enhance_prompt(user_prompt):
    generator = get_prompt_generator()
    if not generator:
        return user_prompt + ", highly detailed, 8k, cinematic lighting"
    
    input_text = f"A beautiful, detailed image of {user_prompt}, featuring"
    try:
        responses = generator(
            input_text, 
            max_length=len(input_text)+150, 
            num_return_sequences=1, 
            truncation=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            temperature=0.9,
            do_sample=True
        )
        generated_text = responses[0]['generated_text']
        clean_text = generated_text.replace(input_text, "").strip()
        if "." in clean_text:
            clean_text = clean_text.split(".")[0]
        return f"{user_prompt}, {clean_text}"
    except Exception as e:
        print(f"Generation Error: {e}")
        return user_prompt

def nltk_process_prompt(prompt):
    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(prompt)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words and len(w) > 2]
        return ", ".join(filtered_sentence) + ", vivid, masterpiece, 8k resolution"
    except Exception:
        return prompt + ", 8k"

def apply_opencv_filter(pil_image, filter_type):
    open_cv_image = np.array(pil_image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    if filter_type == "Grayscale":
        processed_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    elif filter_type == "Canny Edge":
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif filter_type == "Blur":
        processed_image = cv2.GaussianBlur(open_cv_image, (15, 15), 0)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    else:
        return pil_image

    return Image.fromarray(processed_image)

class GenerateRequest(BaseModel):
    prompt: str
    nlp_mode: str = "None"
    filter: str = "None"
    width: int = 1024
    height: int = 1024
    seed: int = -1

@app.post("/generate")
async def generate(req: GenerateRequest):
    status = []
    final_prompt = req.prompt
    
    if req.nlp_mode == "Transformer (Creative)":
        final_prompt = nlp_enhance_prompt(req.prompt)
        status.append(f"Enhanced prompt: {final_prompt}")
    elif req.nlp_mode == "NLTK (Keywords)":
        final_prompt = nltk_process_prompt(req.prompt)
        status.append(f"NLTK optimized: {final_prompt}")

    seed = req.seed if req.seed != -1 else random.randint(0, 1000000)
    url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(final_prompt)}?width={req.width}&height={req.height}&seed={seed}&nologo=true"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            if req.filter != "None":
                image = apply_opencv_filter(image, req.filter)
            
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return {
                "image": f"data:image/jpeg;base64,{img_str}",
                "final_prompt": final_prompt,
                "logs": "\n".join(status)
            }
        else:
            raise HTTPException(status_code=response.status_code, detail="Pollinations AI error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

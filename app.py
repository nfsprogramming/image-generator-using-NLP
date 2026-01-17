import gradio as gr
import requests
import random
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Optional Imports (Heavy NLP) ---
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Transformers/Torch not available: {e}. 'Transformer' mode will be disabled.")
except OSError as e:
    print(f"OS Error loading Transformers/Torch (likely missing VC++ Redist): {e}. 'Transformer' mode will be disabled.")

# --- Setup NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# --- NLP Section: Prompt Generation ---
print("Loading NLP Model...")
prompt_generator = None

# Use a small, fast model for text generation to enhance prompts
if TRANSFORMERS_AVAILABLE:
    try:
        # gpu support if available
        device = 0 if torch.cuda.is_available() else -1
        print(f"Device set to use {'gpu' if device == 0 else 'cpu'}")
        
        # simple text-generation pipeline
        prompt_generator = pipeline(
            "text-generation", 
            model="distilgpt2", 
            device=device
        )
    except Exception as e:
        print(f"Warning: NLP model failed to load: {e}")
        prompt_generator = None
else:
    print("Skipping Transformer model load due to missing dependencies.")

def nlp_enhance_prompt(user_prompt):
    """
    Uses a Transformer model to expand on the user's prompt.
    """
    if not prompt_generator:
        return user_prompt + ", highly detailed, 8k, cinematic lighting"
    
    input_text = f"A beautiful, detailed image of {user_prompt}, featuring"
    
    try:
        # Added repetition_penalty and no_repeat_ngram_size to fix the loop error
        # Also added do_sample and temperature for better variety
        responses = prompt_generator(
            input_text, 
            max_length=len(input_text)+50, 
            num_return_sequences=1, 
            truncation=True,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            temperature=0.8,
            do_sample=True,
            top_k=50,
            top_p=0.9
        )
        generated_text = responses[0]['generated_text']
        
        # Clean up the output: Remove the input prefix and stop at the first period/newline
        clean_text = generated_text.replace(input_text, "").strip()
        if "." in clean_text:
            clean_text = clean_text.split(".")[0]
        
        # Combine and ensure it's not too long
        final_p = f"{user_prompt}, {clean_text}"
        return final_p[:400] # Limit prompt length
    except Exception as e:
        print(f"NLP Generation Error: {e}")
        return user_prompt

def nltk_process_prompt(prompt):
    """
    Uses NLTK to extract key concepts (nouns/adjectives) and remove filler words.
    Good for 'artistic' style prompts which prefer keywords.
    """
    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(prompt)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words and len(w) > 2]
        # Add some artistic styles
        return ", ".join(filtered_sentence) + ", vivid, masterpiece, 8k resolution"
    except Exception as e:
        print(f"NLTK Error: {e}")
        return prompt + ", 8k"

# --- OpenCV Section: Image Processing ---
def apply_opencv_filter(pil_image, filter_type):
    """
    Applies OpenCV filters to the generated image.
    """
    # Convert PIL Image to OpenCV format (numpy array)
    open_cv_image = np.array(pil_image) 
    # PIL is RGB, OpenCV is BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    if filter_type == "Grayscale":
        processed_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        # Convert back to BGR for consistency in saving/display rules later (though gray is 1 ch)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        
    elif filter_type == "Canny Edge":
        # Canny edge detection
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        # Make edges visible (white on black) -> Invert for better look or keep as is? 
        # Let's keep white edges on black
        processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    elif filter_type == "Blur":
        processed_image = cv2.GaussianBlur(open_cv_image, (15, 15), 0)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
    else:
        # No filter
        return pil_image

    # Convert back to PIL Image
    return Image.fromarray(processed_image)

# --- Main Generation Function ---
def generate_and_process(prompt, nlp_mode, filter_choice, width, height, seed):
    """
    Orchestrates the NLP enhancement -> Image Generation -> OpenCV processing flow.
    """
    status = []
    
    # 1. NLP Step
    final_prompt = prompt
    if nlp_mode == "Transformer (Creative)":
        status.append("NLP: Enhancing prompt with Transformer...")
        final_prompt = nlp_enhance_prompt(prompt)
        status.append(f"NLP: Modified prompt to '{final_prompt}'")
    elif nlp_mode == "NLTK (Keywords)":
        status.append("NLP: Optimizing with NLTK...")
        final_prompt = nltk_process_prompt(prompt)
        status.append(f"NLP: Keywords extracted: '{final_prompt}'")
    else:
        status.append(f"Using original prompt: '{final_prompt}'")

    # 2. Generation Step (Pollinations.ai)
    if seed == -1:
        seed = random.randint(0, 1000000)
    
    # Updated URL: Using image.pollinations.ai/prompt/ which is the correct API endpoint for raw images.
    url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(final_prompt)}?width={width}&height={height}&seed={seed}&nologo=true"
    status.append(f"Debug: Requesting URL: {url}")
    
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30)
            status.append(f"Debug: API Status {response.status_code} (Attempt {attempt+1})")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'image' not in content_type:
                    return None, f"Error: API returned {content_type} instead of image.\nContent preview: {response.content[:100]}"
                    
                try:
                    image = Image.open(BytesIO(response.content))
                    status.append("Generation: Success")
                    break # Success, break loop to parse image
                except Exception as img_err:
                    return None, f"Error opening image: {img_err}\nContent preview: {response.content[:100]}"
            
            elif response.status_code in [500, 502, 503, 504]:
                status.append(f"Server Error {response.status_code}, retrying...")
                import time
                time.sleep(1)
                continue # Try again
                
            else:
                return None, "\n".join(status) + f"\nError: API Status {response.status_code}"
                
        except Exception as e:
            status.append(f"Error calling API: {e}")
            if attempt < 2:
                status.append("Retrying...")
                import time
                time.sleep(1)
                continue
            return None, f"Error calling API: {e}"
            
    else:
        # If loop finishes without breaking (failure)
        return None, "\n".join(status) + f"\nFailed after 3 attempts."

    # 3. OpenCV Step
    if filter_choice != "None":
        status.append(f"OpenCV: Applying {filter_choice} filter...")
        # Check if helper is defined or not? It is.
        image = apply_opencv_filter(image, filter_choice)
    
    return image, "\n".join(status)



# --- Gradio UI ---
# --- Custom CSS for Absolute Purity ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

:root {
    --primary: #8b5cf6;
    --primary-hover: #7c3aed;
    --bg-dark: #020617;
    --card-bg: rgba(15, 23, 42, 0.4);
    --border: rgba(255, 255, 255, 0.08); 
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}

/* Global Dark Force */
* {
    border-color: var(--border) !important;
}

html, body, .gradio-container {
    background-color: var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Total Background Purge */
.bg-white, .bg-gray-50, .bg-gray-100, .bg-gray-200, .bg-slate-50, .bg-slate-100,
div[class*="bg-"], section[class*="bg-"], nav[class*="bg-"], 
.block, .form, .prose, .gr-box, .gr-panel, .gr-padded, .gr-form, .gr-input, .gr-compact,
.tab-nav, .tabitem, fieldset, .gr-group, .gr-block, .gr-samples, .gr-samples table {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

/* Fix the Infamous White Boxes (Dropdowns, Buttons, Inputs) */
input, textarea, select, .dropdown, div[class*="wrapper"], div[class*="container"],
button[class*="svelte-"], [class*="gr-input"], [class*="gr-dropdown"],
[data-testid="radio"] div, [data-testid="dropdown"] div, [data-testid="number"] div,
[class*="wrap-"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
}

/* Force Dropdown Selection Box to be dark */
[data-testid="dropdown"] div[class*="select"] {
    background-color: rgba(15, 23, 42, 0.8) !important;
    color: white !important;
}

/* Fix Number Inputs Specifically */
input[type="number"] {
    background-color: rgba(15, 23, 42, 0.8) !important;
    color: white !important;
}

/* Radio Button Styling */
[data-testid="radio"] label {
    background-color: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid var(--border) !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
}
.selected, [class*="selected"], [class*="active"] {
    background-color: var(--primary) !important;
    color: white !important;
}

/* Labels & Block Headers */
[data-testid="block-label"], .label, .gr-panel-header, 
.panel-header, [class*="label-"] {
    background: rgba(31, 41, 55, 0.9) !important;
    color: var(--text-main) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 700 !important;
    padding: 4px 12px !important;
    border-radius: 12px 12px 0 0 !important;
}

/* Table Integration */
tr, td, th {
    background: transparent !important;
    color: var(--text-main) !important;
}
tr:hover {
    background: rgba(139, 92, 246, 0.1) !important;
}

/* Studio Branding */
.main-header {
    margin: 5rem 0 4rem 0;
    text-align: center;
}
.main-header h1 {
    font-size: 5rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #fff 20%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.06em;
}

.glass-panel {
    background: var(--card-bg) !important;
    backdrop-filter: blur(24px) !important;
    border: 1px solid var(--border) !important;
    border-radius: 28px !important;
    padding: 3rem !important;
    box-shadow: 0 60px 120px -30px rgba(0, 0, 0, 1) !important;
}

.generate-btn {
    background: linear-gradient(135deg, var(--primary), #4338ca) !important;
    border: none !important;
    padding: 1.5rem !important;
    border-radius: 20px !important;
    font-weight: 800 !important;
    font-size: 1.4rem !important;
    box-shadow: 0 20px 40px -10px rgba(139, 92, 246, 0.5) !important;
}
.generate-btn:hover {
    transform: translateY(-4px);
    filter: brightness(1.2);
    box-shadow: 0 30px 60px -10px rgba(139, 92, 246, 0.6) !important;
}

/* Output Purification */
.image-preview {
    background: #000 !important;
    border-radius: 20px !important;
    overflow: hidden !important;
    min-height: 550px !important;
}

/* Footer Hide */
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="violet", secondary_hue="slate"), css=custom_css) as demo:
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="main-header"):
            gr.Markdown("# 🎨 NFSI Gen - AI Art Studio")
            gr.Markdown("Purity in Creation. Powered by Neural Language Mastery.")

        with gr.Row():
            with gr.Column(scale=1, elem_classes="glass-panel"):
                prompt_input = gr.Textbox(
                    label="🪄 Your Creative Vision", 
                    placeholder="E.g. A surreal landscape where the sky is made of clockwork gears...",
                    lines=4
                )
                
                with gr.Accordion("🛠️ Neural Styling", open=True):
                    nlp_mode = gr.Radio(
                        label="🧠 Augmentation Logic",
                        choices=["None", "Transformer (Creative)", "NLTK (Keywords)"],
                        value="Transformer (Creative)",
                    )
                    filter_choice = gr.Dropdown(
                        label="✨ Artistic Filters",
                        choices=["None", "Grayscale", "Canny Edge", "Blur"],
                        value="None"
                    )
                    
                with gr.Accordion("⚙️ Technical Build", open=False):
                    with gr.Row():
                        width_slider = gr.Slider(512, 1024, 1024, step=64, label="Width")
                        height_slider = gr.Slider(512, 1024, 1024, step=64, label="Height")
                    seed_input = gr.Number(label="🎲 Seed (-1 for random)", value=-1, precision=0)
                
                generate_btn = gr.Button("🎨 Generate Masterpiece", variant="primary", elem_classes="generate-btn")
                
                gr.Examples(
                    examples=[
                        ["A cyberpunk city in the rain, neon lights, highly detailed, 8k", "Transformer (Creative)", "None"],
                        ["A majestic dragon perched on a mountain top, cinematic lighting", "NLTK (Keywords)", "None"],
                        ["Portrait of a futuristic samurai, digital art style", "Transformer (Creative)", "Blur"],
                        ["Vibrant galaxy inside a glass bottle, cosmic nebula", "None", "None"]
                    ],
                    inputs=[prompt_input, nlp_mode, filter_choice],
                    label="💎 Inspiration Gallery",
                    elem_id="gallery"
                )
            
            with gr.Column(scale=1, elem_classes="glass-panel"):
                image_output = gr.Image(label="🖼️ Final Masterpiece", type="pil", elem_classes="image-preview")
                log_output = gr.Textbox(label="📊 Generation Intelligence", lines=4)

        generate_btn.click(
            fn=generate_and_process,
            inputs=[prompt_input, nlp_mode, filter_choice, width_slider, height_slider, seed_input],
            outputs=[image_output, log_output]
        )

if __name__ == "__main__":
    demo.launch(show_error=True, server_name="127.0.0.1", server_port=7860)

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
import os
import time

# --- Setup NLTK ---
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

# --- NLP ---
def nlp_enhance_prompt(user_prompt):
    additional = ", ultra-detailed digital masterpiece, 8k resolution, cinematic lighting, dramatic atmosphere, intricate textures, sharp focus, masterpiece, highly detailed"
    return user_prompt + additional

def nltk_process_prompt(prompt):
    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(prompt)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words and len(w) > 2]
        return ", ".join(filtered_sentence) + ", vivid, masterpiece, 8k resolution"
    except Exception:
        return prompt + ", 8k"

# --- CV Filters ---
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

def generate_and_process(prompt, nlp_mode, filter_choice, width, height, seed, progress=gr.Progress(track_tqdm=True)):
    if not prompt: return None, "Please define your vision."
    
    final_prompt = prompt
    if nlp_mode == "Transformer":
        final_prompt = nlp_enhance_prompt(prompt)
    elif nlp_mode == "NLTK":
        final_prompt = nltk_process_prompt(prompt)

    seed = int(seed) if seed != -1 else random.randint(0, 1000000)
    url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(final_prompt)}?width={width}&height={height}&seed={seed}&nologo=true"
    
    for attempt in range(3):
        try:
            api_key = os.getenv("POLLINATIONS_API_KEY")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            response = requests.get(url, headers=headers, timeout=120)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                if filter_choice != "None":
                    image = apply_opencv_filter(image, filter_choice)
                short_prompt = final_prompt[:80]
                return image, f"Neural Synthesis Successful | Final Prompt: {short_prompt}..."
            elif response.status_code == 429:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return None, "Pollinations AI is currently under high load. Please try again in secondary."
            else:
                return None, f"Cloud Synthesis Failed: {response.status_code}"
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
            return None, f"Neural Link Error: {str(e)}"

# --- THE ABSOLUTE DARK THEME - VERSION 15 ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root, .dark, body, html, .gradio-container {
    --primary-500: #818cf8 !important;
    --primary-600: #6366f1 !important;
    --bg-color: #020617 !important;
    --block-background-fill: #0f172a !important;
    --block-border-color: rgba(255,255,255,0.1) !important;
    --body-text-color: #f8fafc !important;
    --background-fill-primary: #020617 !important;
    --background-fill-secondary: #020617 !important;
}

/* Force Dark Background everywhere */
body, html, .gradio-container, .main, [class*="bg-"] {
    background-color: #020617 !important;
    background: #020617 !important;
    color: #f8fafc !important;
}

/* Kill all white boxes once and for all */
[class*="svelte-"], .form, .block, .padded, .compact, fieldset, .container, label, .card {
    background-color: transparent !important;
    background: transparent !important;
    border-color: rgba(255,255,255,0.08) !important;
    box-shadow: none !important;
}

/* Header */
.premium-header {
    position: fixed !important; top: 0; left: 0; width: 100%; height: 72px;
    background: rgba(2, 6, 23, 0.9) !important;
    backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    z-index: 100000;
    display: flex !important; justify-content: space-between !important; align-items: center !important;
    padding: 0 40px !important;
}

/* Panels */
.glass-panel {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(40px) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 32px !important;
    padding: 30px !important;
    margin-bottom: 24px !important;
}

/* Input Boxes */
textarea, input, .gr-input, .gr-box {
    background: rgba(0, 0, 0, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    color: white !important;
}

/* Hide Labels */
span[data-testid="block-info"], label > span, .block label span {
    display: none !important;
}

/* Master Button */
.btn-master {
    background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 16px !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
    color: white !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3) !important;
}

/* Radio styling */
[data-testid="radio-group"] {
    background: rgba(255,255,255,0.05) !important;
    padding: 4px !important;
    border-radius: 12px !important;
    display: flex !important;
    gap: 4px !important;
}
[data-testid="radio-group"] label {
    padding: 6px 14px !important;
    border-radius: 8px !important;
}
[aria-checked="true"] {
    background: #6366f1 !important;
    color: white !important;
}

/* Gallery Hover */
#vault-gallery img {
    transition: transform 0.5s ease;
}
#vault-gallery img:hover {
    transform: scale(1.05);
}

footer { display: none !important; }
"""

header_html = """
<div class="premium-header">
    <div style="font-family: 'Outfit', sans-serif; font-weight: 900; font-size: 1.6rem; color: #fff;">NFSI STUDIO</div>
    <div style="display: flex; gap: 40px; font-weight: 700; font-size: 0.85rem; color: #94a3b8;">
        <span style="color: #fff;">DASHBOARD</span>
        <span>GALLERY</span>
        <span>MODELS</span>
    </div>
    <div style="background: rgba(129, 140, 248, 0.15); color: #818cf8; padding: 6px 18px; border-radius: 100px; font-weight: 800; font-size: 0.75rem;">CONNECTED</div>
</div>
"""

head_scripts = """
<script>
// Force Dark Mode immediately
document.documentElement.classList.add('dark');
document.body.classList.add('dark');

window.setPrompt = function(text) {
    const area = document.querySelector('#prompt-area textarea');
    if (area) {
        area.value = text;
        area.dispatchEvent(new Event('input', { bubbles: true }));
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
};
</script>
"""

with gr.Blocks(title="NFSI STUDIO", css=custom_css, head=head_scripts) as demo:
    gr.HTML(header_html)
    
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### 🪄 VISIONARY INPUT")
                
                with gr.Column(elem_id="prompt-area"):
                    prompt_input = gr.Textbox(label="", placeholder="Describe the unseen...", lines=6, container=False)
                    enhance_btn = gr.Button("✨ ENHANCE", elem_classes="btn-master", variant="primary")

                with gr.Accordion("🧠 NEURAL ENGINE", open=True):
                    nlp_mode = gr.Radio(choices=["None", "Transformer", "NLTK"], value="Transformer", label="")
                
                with gr.Accordion("🎨 VISUAL FINISH", open=True):
                    filter_choice = gr.Dropdown(choices=["None", "Grayscale", "Canny Edge", "Blur"], value="None", label="")
                
                with gr.Accordion("⚙️ TECHNICAL", open=False):
                    with gr.Row():
                        w_sld = gr.Slider(512, 1024, 1024, step=64, label="W")
                        h_sld = gr.Slider(512, 1024, 1024, step=64, label="H")
                    sd_num = gr.Number(value=-1, label="Seed")

                generate_btn = gr.Button("🎨 SYNTHESIZE", elem_classes="btn-master")

        with gr.Column(scale=6):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### 🖼️ MANIFESTATION")
                image_out = gr.Image(label="", type="pil", interactive=False, container=False)
                
                with gr.Accordion("📊 PULSE", open=True):
                    log_out = gr.Textbox(label="", lines=2, interactive=False, elem_id="log-output", container=False)
                
                with gr.Row():
                    gr.Button("🔗 EXPORT", variant="secondary")
                    gr.Button("💎 UPSCALE", variant="secondary")
                    gr.Button("🔄 RE-GEN", variant="secondary")

    with gr.Column(elem_classes="glass-panel"):
        gr.Markdown("### 💎 THE VAULT")
        vault_gal = gr.Gallery(
            value=[
                ("assets/gallery/cyberpunk.png", "Cyberpunk Rain City"),
                ("assets/gallery/dragon.png", "Majestic Gold Dragon"),
                ("assets/gallery/samurai.png", "Futuristic Ronin"),
                ("assets/gallery/galaxy.png", "Nebula Crystal")
            ],
            columns=4, rows=1, height=260, allow_preview=False, container=False, show_label=False, elem_id="vault-gallery"
        )

    generate_btn.click(fn=generate_and_process, inputs=[prompt_input, nlp_mode, filter_choice, w_sld, h_sld, sd_num], outputs=[image_out, log_out])
    enhance_btn.click(fn=nlp_enhance_prompt, inputs=[prompt_input], outputs=[prompt_input])
    vault_gal.select(lambda x: x.value['caption'], None, prompt_input)

if __name__ == "__main__":
    gal_path = os.path.abspath("assets/gallery")
    demo.launch(server_name="127.0.0.1", server_port=7860, allowed_paths=[gal_path], theme=gr.themes.Monochrome())

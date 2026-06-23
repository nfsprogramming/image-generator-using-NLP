# LuminaVision

LuminaVision is a premium neural image synthesis studio. It features a sleek, modern React frontend UI that connects to a robust Python/FastAPI backend to generate and enhance AI images.

## Features
- **Neural Image Synthesis**: Instantly generate high-quality images from text prompts using advanced backend integrations (Pollinations AI).
- **Prompt Enhancement**: Leverage `distilgpt2` and `NLTK` engines to automatically expand, clean, and enrich your visual concept descriptions before generation.
- **Style Filters**: Built-in OpenCV post-processing to apply real-time filters (Grayscale, Canny Edge, Gaussian Blur) to your generated artifacts.
- **Real-Time Monitoring**: The frontend dynamically polls backend health and adjusts UI controls based on the Neural Link status.
- **Sleek Interface**: A meticulously designed dark-mode user interface featuring glassmorphic overlays, fluid animations, and custom scrollbars.

## Architecture
- **Frontend**: React, Vite, Tailwind CSS (v4), Axios
- **Backend**: Python 3, FastAPI, Uvicorn, OpenCV, PyTorch, Transformers, NLTK

## Getting Started

### Backend Setup
1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```bash
   python main.py
   ```
   *(The server will run on `http://0.0.0.0:8001` by default)*

### Frontend Setup
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install the Node modules:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *(The application will be accessible at `http://localhost:5174`)*

## Environment Variables
- `POLLINATIONS_API_KEY`: (Optional) Your API key for Pollinations AI image generation.
- `VITE_API_URL`: (Optional) Frontend environment variable to override the default backend URL (`http://localhost:8001`).

## License
Refer to the `LICENSE` file for more details.

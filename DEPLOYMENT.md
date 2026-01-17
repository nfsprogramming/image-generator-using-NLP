# Deployment Guide for AI Art Studio

This project is set up to be deployed in two parts:
1. **Frontend**: Vercel
2. **Backend**: Render

## 1. Backend Deployment (Render)

1.  Push this code to a **GitHub** repository.
2.  Go to [dashboard.render.com](https://dashboard.render.com/).
3.  Click **New +** -> **Web Service**.
4.  Connect your GitHub repository.
5.  **Settings**:
    *   **Runtime**: Python 3
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6.  Click **Create Web Service**.
7.  Once deployed, copy your **Service URL** (e.g., `https://ai-art-studio.onrender.com`).

## 2. Frontend Deployment (Vercel)

1.  Go to [vercel.com](https://vercel.com).
2.  Click **Add New...** -> **Project**.
3.  Import the same GitHub repository.
4.  **Framework Preset**: Vite
5.  **Root Directory**: Click "Edit" and select the `frontend` folder.
6.  **Environment Variables**:
    *   Key: `VITE_API_URL`
    *   Value: `https://image-generator-using-nlp.onrender.com`
    *   *Important: Do not add a trailing slash `/`*
7.  Click **Deploy**.

## 3. Local Development
*   **Backend**: `python main.py` (Runs on port 8001)
*   **Frontend**: `cd frontend && npm run dev` (Runs on port 3000/3005)

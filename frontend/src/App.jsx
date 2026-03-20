import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Wand2, Image as ImageIcon, Sparkles, Layers, Sliders,
  Download, Terminal, AlertCircle, Share2, RefreshCw,
  Maximize2, Cpu, History, Zap, Settings, Command
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";

const PRESETS = [
  { name: "Cyberpunk", prompt: "A cyberpunk city in the rain, neon lights, ultra-detailed masterpiece, 8k", image: "cyberpunk.png" },
  { name: "Fantasy", prompt: "A majestic dragon on a peak, golden scales, dramatic atmosphere, cinematic", image: "dragon.png" },
  { name: "Anime", prompt: "Futuristic samurai portrait, tech armor, intense focus, v-ray render, anime style", image: "samurai.png" },
  { name: "Cosmic", prompt: "Vibrant galaxy inside a crystal bottle, cosmic nebula, sparkling stars, hyperrealism", image: "galaxy.png" }
];

function App() {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const [isEnhancing, setIsEnhancing] = useState(false);

  // Settings
  const [nlpMode, setNlpMode] = useState("Transformer (Creative)");
  const [filter, setFilter] = useState("None");
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [seed, setSeed] = useState(-1);

  const addLog = (msg) => {
    setLogs(curr => [...curr, `[${new Date().toLocaleTimeString()}] ${msg}`].slice(-8));
  };

  const generateImage = async () => {
    if (!prompt) return;

    setLoading(true);
    setError(null);
    setResult(null);
    addLog("Initiating neural synthesis sequence...");

    try {
      const response = await axios.post(`${API_URL}/generate`, {
        prompt,
        nlp_mode: nlpMode,
        filter: filter,
        width,
        height,
        seed: parseInt(seed)
      });

      setResult(response.data);
      addLog("Synthesis complete. Artifact retrieved.");
      if (response.data.logs) {
        response.data.logs.split('\n').forEach(l => addLog(l));
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Connection failed. Ensure backend is running.");
      addLog("CRITICAL: Neural link severed.");
    } finally {
      setLoading(false);
    }
  };

  const enhancePrompt = async () => {
    if (!prompt) return;
    setIsEnhancing(true);
    addLog("Analyzing neural prompt for enhancement...");
    try {
      const response = await axios.post(`${API_URL}/enhance`, { prompt });
      setPrompt(response.data.enhanced);
      addLog("Neural prompt enhanced successfully.");
    } catch (err) {
      console.error(err);
      addLog("WARNING: Neural enhancement failed.");
    } finally {
      setIsEnhancing(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col md:flex-row font-sans">

      {/* Mesh Gradient Background */}
      <div className="mesh-gradient" />

      {/* Sidebar Navigation */}
      <aside className="w-full md:w-20 lg:w-24 bg-dark-950/40 backdrop-blur-2xl border-r border-white/5 flex flex-col items-center py-8 gap-10">
        <div className="w-12 h-12 bg-primary-600 rounded-2xl flex items-center justify-center shadow-lg shadow-primary-500/30">
          <Zap className="text-white fill-white" />
        </div>
        <nav className="flex flex-col gap-8">
          {[Wand2, History, ImageIcon, Settings].map((Icon, i) => (
            <button key={i} className={`p-4 rounded-2xl transition-all duration-300 ${i === 0 ? 'bg-white/10 text-white' : 'text-slate-500 hover:text-white hover:bg-white/5'}`}>
              <Icon className="w-6 h-6" />
            </button>
          ))}
        </nav>
        <div className="mt-auto flex flex-col gap-6 items-center">
          <div className="w-10 h-10 rounded-full border border-white/10 bg-gradient-to-tr from-primary-500 to-indigo-500 shadow-inner" />
        </div>
      </aside>

      {/* Main Studio Area */}
      <main className="flex-1 overflow-y-auto custom-scrollbar p-6 lg:p-10 space-y-10">

        {/* Header */}
        <div className="flex justify-between items-end">
          <header className="space-y-1">
            <h1 className="text-3xl font-black tracking-tight text-white flex items-center gap-3">
              NFSI <span className="opacity-30">/</span> ART STUDIO
            </h1>
            <p className="text-slate-400 font-medium flex items-center gap-2">
              <Cpu className="w-4 h-4 text-primary-500" /> Neural Instance: <span className="text-primary-400">GPT-2-Turbo-4k</span>
            </p>
          </header>
          <div className="hidden lg:flex gap-3">
            <div className="px-4 py-2 glass rounded-xl flex items-center gap-3">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-xs font-bold tracking-widest text-slate-300">SYSTEM ONLINE</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-10">

          {/* Controls Column */}
          <div className="xl:col-span-5 space-y-8">

            {/* Prompt Card */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card relative">
              <label className="label-text">Neural prompt input</label>
              <div className="relative group">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the unseen..."
                  className="input-field min-h-[180px] resize-none text-lg pr-20"
                />
                <button
                  onClick={enhancePrompt}
                  disabled={isEnhancing || !prompt}
                  className={`absolute bottom-4 right-4 p-3 bg-primary-600 rounded-xl transition-all shadow-lg shadow-primary-500/30 group-hover:shadow-primary-500/50 ${isEnhancing ? 'animate-pulse opacity-50' : 'hover:scale-110 active:scale-95'}`}
                  title="Neural Enhancement"
                >
                  <Sparkles className={`w-5 h-5 text-white ${isEnhancing ? 'animate-spin' : ''}`} />
                </button>
              </div>

              {/* Preset Chips */}
              <div className="mt-6 flex flex-wrap gap-2">
                {PRESETS.map((p) => (
                  <button
                    key={p.name}
                    onClick={() => setPrompt(p.prompt)}
                    className="px-4 py-2 rounded-full border border-white/5 bg-white/5 text-[11px] font-bold text-slate-400 hover:border-primary-500/50 hover:text-white transition-all"
                  >
                    #{p.name.toUpperCase()}
                  </button>
                ))}
              </div>
            </motion.div>

            {/* Parameters Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }} className="glass rounded-3xl p-6">
                <label className="label-text">Logic Engine</label>
                <div className="flex flex-col gap-2">
                  {["None", "Transformer", "NLTK"].map((m) => (
                    <button
                      key={m}
                      onClick={() => setNlpMode(m)}
                      className={`px-4 py-3 rounded-xl text-xs font-bold text-left border transition-all ${nlpMode.includes(m) ? 'border-primary-500 bg-primary-500/10 text-white' : 'border-white/5 bg-black/20 text-slate-500 hover:bg-white/5'}`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </motion.div>

              <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }} className="glass rounded-3xl p-6 flex flex-col gap-4">
                <div className="space-y-4">
                  <label className="label-text">Dimensions</label>
                  <div className="flex gap-4">
                    <input type="number" value={width} onChange={(e) => setWidth(e.target.value)} className="input-field text-center py-2" />
                    <Command className="w-4 h-4 mt-3 opacity-20" />
                    <input type="number" value={height} onChange={(e) => setHeight(e.target.value)} className="input-field text-center py-2" />
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="label-text">Style Filter</label>
                  <select value={filter} onChange={(e) => setFilter(e.target.value)} className="input-field py-2 text-xs">
                    {["None", "Grayscale", "Canny Edge", "Blur"].map(f => <option key={f} value={f}>{f}</option>)}
                  </select>
                </div>
              </motion.div>
            </div>

            <button
              onClick={generateImage}
              disabled={loading || !prompt}
              className="btn-primary group"
            >
              {loading ? (
                <RefreshCw className="w-6 h-6 animate-spin" />
              ) : (
                <>
                  <Wand2 className="w-6 h-6 group-hover:rotate-12 transition-transform" />
                  GENERATE ARTIFACT
                </>
              )}
            </button>
          </div>

          {/* Result Column */}
          <div className="xl:col-span-7 space-y-8">

            <motion.div
              layout
              className="glass rounded-[40px] p-2 aspect-square xl:aspect-auto xl:min-h-[700px] flex items-center justify-center relative overflow-hidden group shadow-[0_0_100px_-20px_rgba(129,140,248,0.1)]"
            >
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div key="img" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="w-full h-full relative">
                    <img src={result.image} className="w-full h-full object-contain rounded-[34px]" alt="Gen Art" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-[34px] flex flex-col justify-end p-10">
                      <div className="flex justify-between items-center">
                        <div className="space-y-1">
                          <p className="text-[10px] font-black tracking-widest text-primary-400">METADATA</p>
                          <p className="text-white font-medium line-clamp-1 max-w-md">{result.final_prompt}</p>
                        </div>
                        <div className="flex gap-4">
                          <button className="btn-secondary"><Share2 className="w-5 h-5" /></button>
                          <button className="btn-secondary px-6"><Download className="w-5 h-5" /></button>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center gap-6 opacity-20">
                    <div className="w-24 h-24 rounded-full border-2 border-dashed border-white flex items-center justify-center">
                      <ImageIcon className="w-10 h-10" />
                    </div>
                    <p className="text-xl font-light tracking-widest uppercase">Canvas Void</p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Loading State Overlay */}
              {loading && (
                <div className="absolute inset-0 bg-dark-950/60 backdrop-blur-xl z-50 flex flex-col items-center justify-center space-y-10">
                  <div className="relative">
                    <div className="w-32 h-32 rounded-full border-2 border-white/5 animate-pulse" />
                    <div className="absolute inset-0 border-t-2 border-primary-500 rounded-full animate-spin shadow-[0_0_30px_rgba(129,140,248,0.5)]" />
                    <Sparkles className="absolute inset-x-0 inset-y-0 m-auto w-10 h-10 text-white animate-bounce" />
                  </div>
                  <div className="text-center space-y-2">
                    <h2 className="text-2xl font-black text-white text-glow">SYNTHESIZING...</h2>
                    <p className="text-slate-400 font-mono text-xs uppercase tracking-[0.3em]">Mapping Neural Vectors</p>
                  </div>
                </div>
              )}
            </motion.div>

            {/* Status Console Overlay-style */}
            <div className="glass rounded-[32px] p-8 space-y-4">
              <div className="flex items-center justify-between border-b border-white/5 pb-4">
                <div className="flex items-center gap-3">
                  <Terminal className="text-primary-500 w-5 h-5" />
                  <span className="text-xs font-black tracking-widest text-white">SYSTEM INTELLIGENCE</span>
                </div>
                <div className="px-3 py-1 bg-white/5 rounded-full text-[9px] font-bold text-slate-500 uppercase">Live Output</div>
              </div>
              <div className="space-y-3">
                {logs.length === 0 && <p className="text-slate-600 text-sm font-mono italic">Waiting for connection link...</p>}
                {logs.map((log, i) => (
                  <div key={i} className="flex gap-4 font-mono text-xs">
                    <span className="text-primary-500 opacity-50">#00{i}</span>
                    <span className={`${log.includes('CRITICAL') ? 'text-red-400' : 'text-slate-400'}`}>{log}</span>
                  </div>
                ))}
                {error && <p className="text-red-400 text-xs font-mono bg-red-500/10 p-4 rounded-xl border border-red-500/20">{error}</p>}
              </div>
            </div>

          </div>
        </div>

        {/* Horizontal Inspiration Vault */}
        <section className="space-y-8 pt-10">
          <div className="flex items-center justify-between">
            <h3 className="text-2xl font-black text-white tracking-tight">INSPIRATION VAULT</h3>
            <button className="text-primary-400 text-xs font-bold hover:underline">VIEW ALL COLLECTION</button>
          </div>
          <div className="flex gap-8 overflow-x-auto custom-scrollbar pb-10 -mx-1 px-1">
            {PRESETS.map((p, i) => (
              <motion.div
                key={i}
                whileHover={{ y: -10 }}
                onClick={() => setPrompt(p.prompt)}
                className="min-w-[340px] flex-shrink-0 relative group cursor-pointer"
              >
                <div className="aspect-[16/10] bg-dark-900 rounded-[32px] overflow-hidden border border-white/5 shadow-2xl relative">
                  <img src={`/assets/gallery/${p.image}`} className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" alt={p.name} />
                  <div className={`absolute inset-0 bg-gradient-to-br transition-all duration-700 ${i === 0 ? 'from-purple-500/20' : i === 1 ? 'from-amber-500/20' : i === 2 ? 'from-blue-500/20' : 'from-indigo-500/20'} to-dark-950/60 group-hover:opacity-0 opacity-100`} />
                  <div className="absolute inset-0 flex flex-col justify-end p-8 gap-3 opacity-100 group-hover:bg-dark-950/40 transition-all">
                    <div className="w-10 h-10 glass rounded-xl flex items-center justify-center">
                      <Zap className="w-4 h-4 text-white" />
                    </div>
                    <p className="text-white font-bold text-xl drop-shadow-lg">{p.name}</p>
                    <p className="text-slate-400 text-xs line-clamp-1 opacity-0 group-hover:opacity-100 transition-opacity">{p.prompt}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

      </main>
    </div>
  );
}

export default App;

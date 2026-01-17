import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Wand2, Image as ImageIcon, Sparkles, Layers, Sliders, Download, Terminal, AlertCircle } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";

function App() {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);

  // Settings
  const [nlpMode, setNlpMode] = useState("Transformer (Creative)");
  const [filter, setFilter] = useState("None");
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [seed, setSeed] = useState(-1);

  const generateImage = async () => {
    if (!prompt) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setLogs(["Initializing generation protocol..."]);

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
      setLogs(curr => [...curr, ...response.data.logs.split('\n')]);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Connection failed. Ensure backend is running.");
      setLogs(curr => [...curr, "CRITICAL ERROR: Generation failed."]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-4 md:p-8 lg:p-12 w-full max-w-[2000px] mx-auto flex flex-col gap-12">

      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <div className="inline-flex items-center justify-center p-3 bg-white/5 rounded-2xl mb-4 border border-white/10 ring-1 ring-white/5 relative group">
          <Sparkles className="w-8 h-8 text-primary-500 group-hover:text-amber-400 transition-colors duration-500" />
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse shadow-[0_0_10px_rgba(239,68,68,0.8)]" />
        </div>
        <h1 className="text-5xl md:text-7xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white via-primary-200 to-indigo-300">
          AI Art Studio
        </h1>
        <p className="text-slate-400 text-lg font-light tracking-wide flex items-center justify-center gap-3">
          Neural Language Mastery <span className="text-white/20">•</span> v2.0 <span className="text-white/20">•</span>
          <span className="text-amber-400 font-bold tracking-widest drop-shadow-[0_0_10px_rgba(251,191,36,0.5)]">UNLIMITED</span>
        </p>
      </motion.header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* Left Control Panel */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-4 space-y-6"
        >
          {/* Prompt Section */}
          <div className="glass rounded-3xl p-6 md:p-8 space-y-6">
            <div>
              <label className="label-text flex items-center gap-2">
                <Wand2 className="w-4 h-4" /> Creative Vision
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe your masterpiece..."
                className="input-field min-h-[140px] resize-none text-lg"
              />
            </div>

            {/* NLP Settings */}
            <div className="space-y-4">
              <label className="label-text flex items-center gap-2">
                <Terminal className="w-4 h-4" /> Neural Augmentation
              </label>
              <div className="grid grid-cols-1 gap-2">
                {["None", "Transformer (Creative)", "NLTK (Keywords)"].map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setNlpMode(mode)}
                    className={`px-4 py-3 rounded-xl text-sm font-medium transition-all text-left border ${nlpMode === mode
                      ? 'bg-primary-600/20 border-primary-500 text-white shadow-lg shadow-primary-500/10'
                      : 'bg-dark-950/30 border-white/5 text-slate-400 hover:bg-white/5'
                      }`}
                  >
                    {mode}
                    {mode === nlpMode && <span className="float-right text-primary-400">●</span>}
                  </button>
                ))}
              </div>
            </div>

            {/* Technical Settings */}
            <div className="space-y-4 pt-4 border-t border-white/5">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label-text">Width</label>
                  <input
                    type="number"
                    value={width}
                    onChange={(e) => setWidth(parseInt(e.target.value))}
                    className="input-field"
                  />
                </div>
                <div>
                  <label className="label-text">Height</label>
                  <input
                    type="number"
                    value={height}
                    onChange={(e) => setHeight(parseInt(e.target.value))}
                    className="input-field"
                  />
                </div>
              </div>
              <div>
                <label className="label-text">Seed (-1 Random)</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value))}
                  className="input-field"
                />
              </div>
              <div>
                <label className="label-text flex items-center gap-2">
                  <Layers className="w-4 h-4" /> Filter
                </label>
                <select
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  className="input-field appearance-none bg-dark-950/50"
                >
                  <option value="None">None</option>
                  <option value="Grayscale">Grayscale</option>
                  <option value="Canny Edge">Canny Edge</option>
                  <option value="Blur">Blur</option>
                </select>
              </div>
            </div>

            <button
              onClick={generateImage}
              disabled={loading || !prompt}
              className={`btn-primary flex items-center justify-center gap-2 ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Dreaming...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" /> Generate Masterpiece
                </>
              )}
            </button>
          </div>
        </motion.div>

        {/* Right Output Panel */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="lg:col-span-8 space-y-6"
        >
          <div className="glass rounded-[32px] p-2 min-h-[600px] flex items-center justify-center relative overflow-hidden group">
            {/* Background noise texture or gradient */}
            <div className="absolute inset-0 bg-gradient-to-tr from-dark-950 via-dark-900 to-indigo-950/30 -z-10" />

            <AnimatePresence mode="wait">
              {result ? (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  className="relative w-full h-full"
                >
                  <img
                    src={result.image}
                    alt="Generative Art"
                    className="w-full h-full object-contain rounded-[24px] shadow-2xl"
                  />
                  <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <a
                      href={result.image}
                      download={`nfsi-gen-${Date.now()}.jpg`}
                      className="p-3 bg-black/50 backdrop-blur-md rounded-xl text-white hover:bg-black/70 transition-colors"
                    >
                      <Download className="w-5 h-5" />
                    </a>
                  </div>
                  <div className="absolute bottom-4 left-4 right-4 p-4 bg-black/60 backdrop-blur-md rounded-2xl border border-white/10">
                    <p className="text-xs text-slate-400 uppercase tracking-widest mb-1">Final Prompt</p>
                    <p className="text-sm text-white line-clamp-2">{result.final_prompt}</p>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center space-y-4 text-slate-600"
                >
                  <div className="w-24 h-24 rounded-full bg-white/5 mx-auto flex items-center justify-center border border-white/5">
                    <ImageIcon className="w-10 h-10 opacity-50" />
                  </div>
                  <p className="font-light tracking-wide text-lg">Your canvas awaits.</p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Loading Overlay */}
            {loading && (
              <div className="absolute inset-0 bg-dark-950/80 backdrop-blur-sm z-20 flex flex-col items-center justify-center space-y-6">
                <div className="relative">
                  <div className="w-20 h-20 rounded-full border-4 border-primary-500/20 shadow-[0_0_30px_rgba(139,92,246,0.2)]" />
                  <div className="absolute top-0 left-0 w-20 h-20 rounded-full border-4 border-t-primary-500 animate-spin border-r-transparent border-b-transparent border-l-transparent" />
                  <Sparkles className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-white w-6 h-6 animate-pulse" />
                </div>
                <div className="text-center">
                  <p className="text-white font-medium text-lg">Synthesizing Pixels</p>
                  <p className="text-primary-400 text-sm animate-pulse">Running Neural Inference...</p>
                </div>
              </div>
            )}
          </div>

          {/* Console / Logs */}
          <div className="glass rounded-2xl p-6 overflow-hidden">
            <label className="label-text flex items-center gap-2 mb-4">
              <Terminal className="w-4 h-4" /> System Intelligence
            </label>
            <div className="font-mono text-sm space-y-2 max-h-40 overflow-y-auto text-slate-400 custom-scrollbar">
              {logs.length === 0 && <span className="opacity-50">System ready. Waiting for input...</span>}
              {logs.map((log, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex gap-2"
                >
                  <span className="text-primary-500/50">{">"}</span>
                  <span>{log}</span>
                </motion.div>
              ))}
              {error && (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex gap-2 text-red-400"
                >
                  <AlertCircle className="w-4 h-4 mt-0.5" />
                  <span>{error}</span>
                </motion.div>
              )}
            </div>
          </div>

        </motion.div>
      </div>
    </div>
  );
}

export default App;

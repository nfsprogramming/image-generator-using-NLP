import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";

function App() {
  const [prompt, setPrompt] = useState("A cyberpunk city in the rain, neon lights, ultra-detailed masterpiece, 8k");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [isOnline, setIsOnline] = useState(false);

  // Settings
  const [nlpMode, setNlpMode] = useState("Transformer");
  const [filter, setFilter] = useState("None");
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [seed, setSeed] = useState(-1);

  const logsEndRef = useRef(null);

  const addLog = (msg, type = "normal") => {
    setLogs(curr => [...curr, { time: new Date().toLocaleTimeString(), msg, type }].slice(-20));
  };

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        await axios.get(`${API_URL}/health`, { timeout: 2000 });
        setIsOnline(true);
      } catch (err) {
        setIsOnline(false);
      }
    };
    checkConnection();
    const interval = setInterval(checkConnection, 5000);
    return () => clearInterval(interval);
  }, []);

  const generateImage = async () => {
    if (!prompt) return;

    setLoading(true);
    setError(null);
    setResult(null);
    addLog("Initiating neural synthesis sequence...", "highlight");

    let apiNlpMode = "None";
    if (nlpMode === "Transformer") apiNlpMode = "Transformer (Creative)";
    if (nlpMode === "NLTK") apiNlpMode = "NLTK (Keywords)";

    try {
      const response = await axios.post(`${API_URL}/generate`, {
        prompt,
        nlp_mode: apiNlpMode,
        filter: filter,
        width: parseInt(width),
        height: parseInt(height),
        seed: parseInt(seed)
      });

      setResult(response.data);
      addLog("Synthesis complete. Artifact retrieved.", "success");
      if (response.data.logs) {
        response.data.logs.split('\n').forEach(l => addLog(l));
      }
    } catch (err) {
      console.error(err);
      const errMsg = err.response?.data?.detail || "Connection failed. Ensure backend is running.";
      setError(errMsg);
      addLog(`CRITICAL: Neural link severed - ${errMsg}`, "error");
    } finally {
      setLoading(false);
    }
  };

  const enhancePrompt = async () => {
    if (!prompt) return;
    setIsEnhancing(true);
    addLog("Analyzing neural prompt for enhancement...", "highlight");
    try {
      const response = await axios.post(`${API_URL}/enhance`, { prompt });
      setPrompt(response.data.enhanced);
      addLog("Neural prompt enhanced successfully.", "success");
    } catch (err) {
      console.error(err);
      addLog("WARNING: Neural enhancement failed.", "error");
    } finally {
      setIsEnhancing(false);
    }
  };

  const downloadImage = () => {
    if (!result?.image) return;
    const a = document.createElement('a');
    a.href = result.image;
    a.download = `lumina-artifact-${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    addLog("Artifact downloaded successfully.", "success");
  };

  return (
    <div className="min-h-screen font-sans">
      <main className="min-h-screen flex flex-col">
        {/* BEGIN: Top Header */}
        <header className="h-16 shrink-0 flex items-center justify-between px-8 bg-studio-dark/50 border-b border-white/5">
          <div className="flex items-center gap-6">
            <h1 className="text-xl font-bold tracking-[0.2em] text-white uppercase"><span className="text-studio-accent font-black">LUMINA</span>VISION</h1>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-studio-accent/10 rounded-full border border-studio-accent/30 shadow-[0_0_10px_rgba(99,102,241,0.2)]">
              <div className="w-1.5 h-1.5 rounded-full bg-studio-accent animate-pulse"></div>
              <span className="text-[10px] font-bold tracking-widest text-studio-accent uppercase">Neural Engine <span className="text-white ml-1">GPT-2-Turbo</span></span>
            </div>
          </div>
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${isOnline ? 'bg-green-500/10 border-green-500/20' : 'bg-red-500/10 border-red-500/20'}`}>
            <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></span>
            <span className={`text-[10px] font-bold tracking-widest uppercase ${isOnline ? 'text-green-500' : 'text-red-500'}`}>{isOnline ? 'System Online' : 'System Offline'}</span>
          </div>
        </header>
        {/* END: Top Header */}

        {/* BEGIN: Workspace */}
        <div className="flex-1 flex p-6 gap-6">
          {/* Left Controls Column */}
          <div className="w-[440px] flex flex-col gap-6 pr-2">
            {/* Neural Prompt Card */}
            <section className="bg-studio-panel panel-border rounded-xl p-5 flex flex-col">
              <label className="text-[10px] font-bold tracking-[0.2em] text-studio-accent uppercase mb-3">Neural Prompt Input</label>
              <div className="relative bg-studio-dark/50 rounded-lg p-4 border border-white/5 focus-within:border-studio-accent/50 transition-colors">
                <textarea 
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="w-full h-32 bg-transparent border-none focus:ring-0 text-studio-highlight placeholder-white/20 resize-none font-sans leading-relaxed outline-none" 
                  placeholder="Enter your visual concept description here..."
                ></textarea>
                <div className="absolute bottom-3 right-3 flex gap-2">
                  <button onClick={() => setPrompt("")} className="p-1.5 bg-studio-accent/20 text-studio-accent rounded hover:bg-studio-accent/30 transition-colors" title="Clear Prompt">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1a1 1 0 112 0v1a1 1 0 11-2 0zM13.536 14.95a1 1 0 010-1.414l.707-.707a1 1 0 011.414 1.414l-.707.707a1 1 0 01-1.414 0zM6.464 14.95l.707-.707a1 1 0 10-1.414-1.414l-.707.707a1 1 0 001.414 1.414z"></path></svg>
                  </button>
                  <button onClick={enhancePrompt} disabled={isEnhancing || !prompt} className="p-1.5 bg-studio-accent text-white rounded hover:bg-studio-glow transition-colors disabled:opacity-50" title="Enhance Prompt">
                    <svg className={`w-4 h-4 ${isEnhancing ? 'animate-spin' : ''}`} fill="currentColor" viewBox="0 0 20 20"><path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM11 13a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
                  </button>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 mt-4">
                <span onClick={() => setPrompt("A cyberpunk city in the rain, neon lights, ultra-detailed masterpiece, 8k")} className="px-2 py-1 text-[10px] font-bold bg-white/5 border border-white/10 rounded uppercase hover:border-studio-accent/50 cursor-pointer transition-colors">#Cyberpunk</span>
                <span onClick={() => setPrompt("A majestic dragon on a peak, golden scales, dramatic atmosphere, cinematic")} className="px-2 py-1 text-[10px] font-bold bg-white/5 border border-white/10 rounded uppercase hover:border-studio-accent/50 cursor-pointer transition-colors">#Fantasy</span>
                <span onClick={() => setPrompt("Futuristic samurai portrait, tech armor, intense focus, v-ray render, anime style")} className="px-2 py-1 text-[10px] font-bold bg-white/5 border border-white/10 rounded uppercase hover:border-studio-accent/50 cursor-pointer transition-colors">#Anime</span>
                <span onClick={() => setPrompt("Vibrant galaxy inside a crystal bottle, cosmic nebula, sparkling stars, hyperrealism")} className="px-2 py-1 text-[10px] font-bold bg-white/5 border border-white/10 rounded uppercase hover:border-studio-accent/50 cursor-pointer transition-colors">#Cosmic</span>
              </div>
            </section>

            {/* Configuration Grid */}
            <div className="grid grid-cols-2 gap-4">
              {/* Logic Engine Selection */}
              <section className="bg-studio-panel panel-border rounded-xl p-4">
                <label className="text-[10px] font-bold tracking-[0.2em] text-studio-accent uppercase block mb-3">Logic Engine</label>
                <div className="space-y-2">
                  {["None", "Transformer", "NLTK"].map((mode) => (
                    <div 
                      key={mode}
                      onClick={() => setNlpMode(mode)}
                      className={`px-3 py-2 text-xs rounded border cursor-pointer transition-colors ${nlpMode === mode ? 'border-studio-accent bg-studio-accent/10 text-white' : 'border-white/5 bg-studio-dark/30 hover:bg-white/5'}`}
                    >
                      {mode}
                    </div>
                  ))}
                </div>
              </section>

              {/* Dimensions & Style Filter */}
              <div className="space-y-4">
                <section className="bg-studio-panel panel-border rounded-xl p-4">
                  <label className="text-[10px] font-bold tracking-[0.2em] text-studio-accent uppercase block mb-3">Dimensions</label>
                  <div className="flex items-center gap-2">
                    <input type="number" value={width} onChange={(e) => setWidth(e.target.value)} className="w-full bg-studio-dark/50 border border-white/5 text-xs text-center rounded py-2 focus:ring-1 focus:ring-studio-accent focus:border-studio-accent focus:outline-none" />
                    <span className="text-white/20 text-xs">×</span>
                    <input type="number" value={height} onChange={(e) => setHeight(e.target.value)} className="w-full bg-studio-dark/50 border border-white/5 text-xs text-center rounded py-2 focus:ring-1 focus:ring-studio-accent focus:border-studio-accent focus:outline-none" />
                  </div>
                </section>
                
                <section className="bg-studio-panel panel-border rounded-xl p-4">
                  <label className="text-[10px] font-bold tracking-[0.2em] text-studio-accent uppercase block mb-3">Style Filter</label>
                  <select value={filter} onChange={(e) => setFilter(e.target.value)} className="w-full bg-studio-dark/50 border border-white/5 text-xs rounded py-2 px-3 focus:ring-1 focus:ring-studio-accent focus:border-studio-accent outline-none">
                    {["None", "Grayscale", "Canny Edge", "Blur"].map(f => <option key={f} value={f} className="bg-studio-dark text-white">{f}</option>)}
                  </select>
                </section>
              </div>
            </div>

            {/* Action Button */}
            <button 
              onClick={generateImage} 
              disabled={loading || !prompt || !isOnline}
              className="w-full py-4 bg-gradient-to-r from-studio-accent to-studio-glow text-white font-bold tracking-widest rounded-xl flex items-center justify-center gap-3 btn-glow disabled:opacity-50 transition-all"
            >
              {loading ? (
                <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
              )}
              {loading ? "SYNTHESIZING..." : !isOnline ? "WAITING FOR BACKEND..." : "GENERATE ARTIFACT"}
            </button>
          </div>

          {/* Main Canvas Display */}
          <div className="flex-1 flex flex-col gap-6 min-h-[600px]">
            {/* Canvas Void / Result */}
            <section className="flex-1 bg-studio-panel panel-border rounded-2xl canvas-dots relative overflow-hidden flex items-center justify-center group">
              <div className="absolute inset-0 bg-gradient-to-t from-studio-dark/20 to-transparent pointer-events-none z-0"></div>
              
              {result ? (
                <div className="w-full h-full relative z-10 flex items-center justify-center p-8">
                  <img src={result.image} alt="Generated Art" className="max-w-full max-h-full object-contain rounded-xl shadow-2xl shadow-studio-glow/20" />
                  <div className="absolute bottom-6 left-6 right-6 p-4 bg-studio-dark/80 backdrop-blur border border-white/10 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity flex justify-between items-center gap-4">
                    <div className="min-w-0 flex-1">
                      <p className="text-[10px] font-bold text-studio-accent tracking-widest uppercase mb-1">Final Prompt Config</p>
                      <p className="text-sm text-white/90 truncate">{result.final_prompt}</p>
                    </div>
                    <button onClick={downloadImage} className="shrink-0 p-3 bg-studio-accent/20 hover:bg-studio-accent text-white border border-studio-accent/50 rounded-lg transition-colors flex items-center justify-center" title="Download Image">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center space-y-4 z-10 opacity-50 group-hover:opacity-80 transition-opacity">
                  <div className="mx-auto w-16 h-16 rounded-full border-2 border-dashed border-studio-text flex items-center justify-center">
                    {loading ? (
                      <svg className="w-8 h-8 animate-spin text-studio-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
                    ) : (
                      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
                    )}
                  </div>
                  <p className="text-sm font-bold tracking-[0.4em] uppercase">{loading ? 'Synthesizing' : 'Canvas Void'}</p>
                </div>
              )}

              {/* Decorative corner accents */}
              <div className="absolute top-4 left-4 w-4 h-4 border-l-2 border-t-2 border-white/10"></div>
              <div className="absolute top-4 right-4 w-4 h-4 border-r-2 border-t-2 border-white/10"></div>
              <div className="absolute bottom-4 left-4 w-4 h-4 border-l-2 border-b-2 border-white/10"></div>
              <div className="absolute bottom-4 right-4 w-4 h-4 border-r-2 border-b-2 border-white/10"></div>
            </section>

            {/* System Logs Console */}
            <section className="h-48 bg-studio-panel panel-border rounded-xl p-5 flex flex-col">
              <header className="flex items-center justify-between mb-4 border-b border-white/5 pb-3">
                <div className="flex items-center gap-3">
                  <svg className="w-4 h-4 text-studio-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
                  <h2 className="text-[10px] font-bold tracking-[0.2em] uppercase text-white">System Intelligence</h2>
                </div>
                <div className="px-2 py-0.5 bg-studio-accent/10 border border-studio-accent/20 rounded-sm">
                  <span className="text-[8px] font-bold text-studio-accent tracking-tighter">LIVE OUTPUT</span>
                </div>
              </header>
              <div className="flex-1 overflow-y-auto font-mono text-[11px] space-y-2 text-studio-text/80 pr-2">
                {logs.length === 0 ? (
                  <div className="flex gap-4 opacity-50">
                    <span className="text-white/20">#000</span>
                    <span className="text-studio-accent">[{new Date().toLocaleTimeString()}]</span>
                    <span>Ready for user input...</span>
                  </div>
                ) : (
                  logs.map((log, i) => (
                    <div key={i} className="flex gap-4">
                      <span className="text-white/20">#{String(i).padStart(3, '0')}</span>
                      <span className={`${log.type === 'error' ? 'text-red-400' : 'text-studio-accent'}`}>[{log.time}]</span>
                      <span className={`${log.type === 'error' ? 'text-red-400' : log.type === 'highlight' ? 'text-white' : log.type === 'success' ? 'text-green-400' : ''}`}>{log.msg}</span>
                    </div>
                  ))
                )}
                <div ref={logsEndRef} />
              </div>
            </section>
          </div>
        </div>
        {/* END: Workspace */}

        {/* BEGIN: Inspiration Vault */}
        <section className="shrink-0 flex flex-col bg-studio-dark/50 border-t border-white/5">
          <header className="h-12 flex items-center justify-between px-8">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold tracking-[0.2em] text-white uppercase">Inspiration Vault</span>
            </div>
            <button className="text-[10px] font-bold tracking-widest text-studio-accent uppercase hover:text-studio-glow transition-colors">
              View All Collection
            </button>
          </header>
          
          <div className="px-8 pb-6">
            <div className="flex gap-4 overflow-x-auto pb-4 custom-scrollbar">
              {[1, 2, 3, 4, 5, 6, 7].map(i => (
                <div key={i} className="min-w-[150px] h-[100px] rounded-lg overflow-hidden relative group cursor-pointer border border-white/5 hover:border-studio-accent transition-colors">
                  <img src={`https://picsum.photos/seed/${i * 123}/300/200`} alt={`Inspiration ${i}`} className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity duration-300" />
                  <div className="absolute inset-0 bg-gradient-to-t from-studio-dark/90 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end p-3">
                    <span className="text-[10px] text-white tracking-widest uppercase font-bold">Use Style</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
        {/* END: Inspiration Vault */}
      </main>
    </div>
  );
}

export default App;

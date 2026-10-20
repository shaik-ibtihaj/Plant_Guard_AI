import { ChangeEvent, CSSProperties, DragEvent, FormEvent, ReactNode, useEffect, useRef, useState } from 'react';
import './styles.css';

type PredictionCandidate = { class_name: string; class_index: number; confidence: number };
type Prediction = {
  record_id: number; model: string; predicted_class: string; confidence: number;
  top_predictions: PredictionCandidate[]; image_width: number; image_height: number;
  inference_time_ms: number;
};
type Explanation = {
  explained_class: string; overlay_png_base64: string; confidence: number;
};
type HistoryRecord = {
  id: number; created_at: string; operation: string; model_name: string;
  predicted_class: string; confidence: number; result: Record<string, unknown>;
};
type Notice = { tone: 'error' | 'success'; text: string } | null;

const API_KEY_STORAGE = 'plant-guard-api-key';
const confidencePercent = (value: number) => `${(value * 100).toFixed(1)}%`;
const readableClass = (value: string) => value.replace(/___/g, ' · ').replace(/_/g, ' ');

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, ReactNode> = {
    leaf: <><path d="M20.8 3.2C11.5 2.6 5.2 5.1 4.1 11c-.7 3.8 2.2 6.1 5.6 5.4 5.9-1.1 8.4-7.4 11.1-13.2Z"/><path d="M3 21c3.4-5.5 7.3-9.3 13-13"/></>,
    scan: <><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="3"/></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/></>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    arrow: <><path d="M5 12h14M13 6l6 6-6 6"/></>,
    upload: <><path d="M12 16V4m-5 5 5-5 5 5"/><path d="M20 16.5v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2"/></>,
    close: <><path d="m18 6-12 12M6 6l12 12"/></>,
    check: <><path d="m5 12 4 4L19 6"/></>,
    shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="m9 12 2 2 4-4"/></>,
    sprout: <><path d="M12 22v-9"/><path d="M12 13C5 13 4 8 4 4c5 0 9 2 9 8M12 16c0-5 3-8 8-8 0 4-1 9-8 9"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] ?? paths.leaf}</svg>;
}

function App() {
  const [activeTab, setActiveTab] = useState<'diagnose' | 'severity' | 'history'>('diagnose');
  const [apiKey, setApiKey] = useState(() => localStorage.getItem(API_KEY_STORAGE) ?? '');
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [severity, setSeverity] = useState<Record<string, unknown> | null>(null);
  const [leafMask, setLeafMask] = useState<File | null>(null);
  const [lesionMask, setLesionMask] = useState<File | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState<Notice>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'ready' | 'offline'>('checking');
  const imageInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch('/health/ready').then((response) => setApiStatus(response.ok ? 'ready' : 'offline')).catch(() => setApiStatus('offline'));
  }, []);

  useEffect(() => {
    if (!image) { setPreview(''); return; }
    const url = URL.createObjectURL(image);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  useEffect(() => {
    if (activeTab === 'history') void loadHistory();
  }, [activeTab]);

  const request = async (path: string, init: RequestInit = {}) => {
    const headers = new Headers(init.headers);
    if (apiKey.trim()) headers.set('X-API-Key', apiKey.trim());
    const response = await fetch(path, { ...init, headers });
    if (!response.ok) {
      const body = await response.json().catch(() => null) as { detail?: string } | null;
      throw new Error(body?.detail || `Request failed (${response.status})`);
    }
    return response.status === 204 ? null : response.json();
  };

  const setSelectedImage = (file?: File) => {
    setNotice(null); setPrediction(null); setExplanation(null); setSeverity(null);
    if (!file) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setNotice({ tone: 'error', text: 'Choose a JPEG, PNG, or WEBP image.' }); return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setNotice({ tone: 'error', text: 'Image must be smaller than 10 MB.' }); return;
    }
    setImage(file); setLeafMask(null); setLesionMask(null);
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault(); setSelectedImage(event.dataTransfer.files[0]);
  };
  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => setSelectedImage(event.target.files?.[0]);

  const analyze = async (event: FormEvent) => {
    event.preventDefault(); if (!image) return;
    setBusy('predict'); setNotice(null); setPrediction(null); setExplanation(null);
    const body = new FormData(); body.append('file', image);
    try {
      const result = await request('/api/v1/predict?top_k=5', { method: 'POST', body }) as Prediction;
      setPrediction(result); setNotice({ tone: 'success', text: 'Analysis complete. Review the result as an advisory estimate.' });
    } catch (error) { setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Unable to analyze this image.' }); }
    finally { setBusy(''); }
  };

  const explain = async () => {
    if (!image) return; setBusy('explain'); setNotice(null);
    const body = new FormData(); body.append('file', image);
    try { setExplanation(await request('/api/v1/explain', { method: 'POST', body }) as Explanation); }
    catch (error) { setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Could not create an explanation.' }); }
    finally { setBusy(''); }
  };

  const assessSeverity = async (event: FormEvent) => {
    event.preventDefault();
    if (!image || !leafMask || !lesionMask) return;
    setBusy('severity'); setNotice(null);
    const body = new FormData(); body.append('image', image); body.append('leaf_mask', leafMask); body.append('lesion_mask', lesionMask);
    try { setSeverity(await request('/api/v1/severity', { method: 'POST', body }) as Record<string, unknown>); setNotice({ tone: 'success', text: 'Area assessment complete.' }); }
    catch (error) { setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Could not assess severity.' }); }
    finally { setBusy(''); }
  };

  const loadHistory = async () => {
    setBusy('history');
    try { const result = await request('/api/v1/records?limit=30&offset=0') as { items: HistoryRecord[] }; setHistory(result.items); }
    catch (error) { setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Could not load history.' }); }
    finally { setBusy(''); }
  };

  const deleteHistoryRecord = async (id: number) => {
    try { await request(`/api/v1/records/${id}`, { method: 'DELETE' }); setHistory((items) => items.filter((item) => item.id !== id)); }
    catch (error) { setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Could not remove record.' }); }
  };

  const saveKey = () => {
    if (apiKey.trim()) localStorage.setItem(API_KEY_STORAGE, apiKey.trim());
    else localStorage.removeItem(API_KEY_STORAGE);
    setNotice({ tone: 'success', text: 'API key preference saved on this device.' });
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home" onClick={() => setActiveTab('diagnose')}><span className="brand-mark"><Icon name="leaf" size={23} /></span><span>plant<span className="brand-light">guard</span><small>FIELD INTELLIGENCE</small></span></a>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          <button className={activeTab === 'diagnose' ? 'active' : ''} onClick={() => setActiveTab('diagnose')}><Icon name="scan" />Diagnose plant</button>
          <button className={activeTab === 'severity' ? 'active' : ''} onClick={() => setActiveTab('severity')}><Icon name="layers" />Severity assessment</button>
          <button className={activeTab === 'history' ? 'active' : ''} onClick={() => setActiveTab('history')}><Icon name="clock" />Analysis history</button>
        </nav>
        <div className="sidebar-bottom">
          <div className="privacy-card"><Icon name="shield" size={18} /><div><strong>Private by design</strong><span>Images are analyzed in memory and never saved.</span></div></div>
          <div className="sidebar-foot"><span className={`status-dot ${apiStatus}`} />API {apiStatus === 'ready' ? 'connected' : apiStatus === 'checking' ? 'checking' : 'unavailable'}<span className="version">v0.2</span></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar"><div className="breadcrumb">Plant Guard <span>/</span> {activeTab === 'diagnose' ? 'New analysis' : activeTab === 'severity' ? 'Severity assessment' : 'History'}</div><button className="key-button" onClick={() => document.getElementById('api-key')?.focus()}><Icon name="shield" size={17} /> API access</button></header>
        <div className="content-wrap">
          {activeTab !== 'history' && <div className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line" />PLANT HEALTH, WITH CLARITY</div><h1>{activeTab === 'diagnose' ? <>A clearer view of<br /><em>plant health.</em></> : <>Measure what<br /><em>matters most.</em></>}</h1><p>{activeTab === 'diagnose' ? 'Upload a leaf photo for an AI-powered disease estimate and visual explanation.' : 'Quantify visible damage with aligned leaf and lesion masks.'}</p></div><div className="heading-art"><div className="art-ring ring-one" /><div className="art-ring ring-two" /><span className="art-leaf"><Icon name="leaf" size={73} /></span><span className="art-spark spark-a">✳</span><span className="art-spark spark-b">·</span></div></div>}

          {notice && <div className={`notice ${notice.tone}`} role="status"><Icon name={notice.tone === 'success' ? 'check' : 'close'} size={17} />{notice.text}<button aria-label="Dismiss message" onClick={() => setNotice(null)}><Icon name="close" size={15} /></button></div>}

          {activeTab === 'diagnose' && <section className="analysis-grid">
            <form className="upload-card card" onSubmit={analyze}>
              <div className="card-topline"><div><span className="step-number">01</span><h2>Upload a leaf image</h2></div><span className="file-limit">MAX 10 MB</span></div>
              <input ref={imageInput} className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={onFileChange} />
              {!preview ? <div className="drop-zone" onDragOver={(event) => event.preventDefault()} onDrop={onDrop} onClick={() => imageInput.current?.click()} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') imageInput.current?.click(); }}><span className="upload-icon"><Icon name="upload" size={22} /></span><strong>Drop your image here</strong><span>or <b>browse files</b> from your device</span><small>JPEG, PNG or WEBP</small></div> : <div className="preview-box"><img src={preview} alt="Selected plant leaf" /><button type="button" className="remove-image" onClick={() => { setImage(null); setPrediction(null); }} aria-label="Remove image"><Icon name="close" size={16} /></button><span className="preview-name">{image?.name}</span></div>}
              <div className="upload-hint"><span className="hint-icon">i</span>For best results, use a sharp, well-lit close-up of one leaf.</div>
              <button className="primary-button" type="submit" disabled={!image || Boolean(busy)}>{busy === 'predict' ? <><span className="spinner" />Analyzing image…</> : <>Analyze image <Icon name="arrow" size={18} /></>}</button>
              <div className="secure-note"><Icon name="shield" size={14} />Your image is not stored</div>
            </form>

            <div className="result-column">
              {!prediction ? <div className="empty-result card"><div className="empty-orbit"><Icon name="sprout" size={34} /></div><span className="eyebrow">YOUR ANALYSIS</span><h2>Insight starts<br />with one leaf.</h2><p>Your prediction, confidence, and visual explanation will appear here.</p><div className="empty-foot"><span>01</span> Upload an image <i /> <span>02</span> Review insight</div></div> : <div className="result-card card">
                <div className="result-label"><span className="result-check"><Icon name="check" size={15} /></span>ANALYSIS COMPLETE<span className="result-time">{prediction.inference_time_ms.toFixed(0)} ms</span></div>
                <div className="prediction-main"><div><span className="field-label">LIKELY IDENTIFICATION</span><h2>{readableClass(prediction.predicted_class)}</h2></div><div className="confidence-ring" style={{ '--confidence': `${prediction.confidence * 100}%` } as CSSProperties}><div><strong>{confidencePercent(prediction.confidence)}</strong><span>confidence</span></div></div></div>
                <div className="confidence-note">Model confidence is an estimate, not a measure of disease severity.</div>
                <div className="top-list-title">OTHER POSSIBLE MATCHES</div>
                {prediction.top_predictions.slice(1).map((candidate) => <div className="candidate" key={candidate.class_index}><span>{readableClass(candidate.class_name)}</span><div className="candidate-track"><i style={{ width: `${Math.max(candidate.confidence * 100, 2)}%` }} /></div><b>{confidencePercent(candidate.confidence)}</b></div>)}
                <div className="result-actions"><button className="secondary-button" onClick={explain} disabled={Boolean(busy)}>{busy === 'explain' ? <span className="spinner dark" /> : <Icon name="scan" size={17} />}Show visual explanation</button><button className="text-button" onClick={() => { setPrediction(null); setExplanation(null); }}>New analysis</button></div>
                {explanation && <div className="explanation-panel"><div><strong>Visual explanation</strong><span>Highlighted regions influenced the model's {readableClass(explanation.explained_class)} estimate.</span></div><img src={`data:image/png;base64,${explanation.overlay_png_base64}`} alt="Grad-CAM attribution overlay" /></div>}
                <div className="advisory"><span>!</span><p><strong>Use as a starting point.</strong> Confirm with a local plant specialist before treating. This model has not been validated for field diagnosis.</p></div>
              </div>}
            </div>
          </section>}

          {activeTab === 'severity' && <section className="severity-layout"><form className="card severity-card" onSubmit={assessSeverity}><div className="card-topline"><div><span className="step-number">01</span><h2>Provide aligned masks</h2></div><span className="file-limit">SAME SIZE</span></div><p className="section-intro">Use the same source image and two binary masks. White pixels mark the leaf and lesion; black pixels mark the background.</p><div className="source-summary">{preview ? <img src={preview} alt="Selected source leaf" /> : <span className="source-placeholder"><Icon name="leaf" size={22} /></span>}<div><strong>{image?.name ?? 'No source image selected'}</strong><span>JPEG, PNG or WEBP · up to 10 MB</span></div><button type="button" className="small-link" onClick={() => imageInput.current?.click()}>Choose</button></div><input ref={imageInput} className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={onFileChange} />
              {[['Leaf mask', leafMask, setLeafMask, 'Silhouette of the full leaf'], ['Lesion mask', lesionMask, setLesionMask, 'White areas mark visible damage']].map(([label, file, setter, help]) => <label className="mask-input" key={label as string}><span><strong>{label as string}</strong><small>{help as string}</small></span><span className="mask-pick">{(file as File | null)?.name ?? 'Choose file'}<input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => (setter as (file: File | null) => void)(event.target.files?.[0] ?? null)} /></span></label>)}
              <button className="primary-button" disabled={!image || !leafMask || !lesionMask || Boolean(busy)}>{busy === 'severity' ? <><span className="spinner" />Measuring area…</> : <>Assess affected area <Icon name="arrow" size={18} /></>}</button><div className="secure-note"><Icon name="shield" size={14} />Images and masks are not stored</div></form>
            <div className="card severity-result">{severity ? <><span className="eyebrow">AREA ASSESSMENT</span><h2>{String(severity.severity)}</h2><div className="severity-meter"><i style={{ width: `${Number(severity.affected_area_ratio) * 100}%` }} /></div><div className="area-stat"><strong>{(Number(severity.affected_area_ratio) * 100).toFixed(1)}%</strong><span>of measured leaf area affected</span></div><p>{String(severity.recommendation)}</p><div className="severity-foot">Based on supplied pixel masks · {Number(severity.affected_pixels).toLocaleString()} affected pixels</div></> : <><div className="empty-orbit"><Icon name="layers" size={32} /></div><span className="eyebrow">AREA, NOT CONFIDENCE</span><h2>A measured view<br />of visible damage.</h2><p>Upload matching masks to estimate how much of the leaf area is affected.</p><div className="mask-legend"><span><i className="legend-leaf" />Leaf area</span><span><i className="legend-lesion" />Lesion area</span></div></>}</div></section>}

          {activeTab === 'history' && <section className="history-page"><div className="history-heading"><div><div className="eyebrow"><span className="eyebrow-line" />YOUR WORKSPACE</div><h1>Analysis <em>history.</em></h1><p>Recent results are saved as metadata only. Uploaded images are never retained.</p></div><button className="secondary-button" onClick={() => void loadHistory()}><Icon name="clock" size={17} />Refresh</button></div><div className="card history-card"><div className="history-table-head"><span>RESULT</span><span>TYPE</span><span>CONFIDENCE</span><span>DATE</span><span /></div>{busy === 'history' ? <div className="history-empty">Loading recent analyses…</div> : history.length === 0 ? <div className="history-empty"><div className="empty-orbit"><Icon name="clock" size={28} /></div><strong>No analysis history yet</strong><span>When you analyze a plant, the result will appear here.</span><button className="text-button" onClick={() => setActiveTab('diagnose')}>Start an analysis <Icon name="arrow" size={15} /></button></div> : history.map((item) => <div className="history-row" key={item.id}><strong>{readableClass(item.predicted_class)}</strong><span className="operation-tag">{item.operation}</span><b>{confidencePercent(item.confidence)}</b><time>{new Date(item.created_at).toLocaleString()}</time><button className="icon-button" onClick={() => void deleteHistoryRecord(item.id)} aria-label={`Delete record ${item.id}`}><Icon name="close" size={16} /></button></div>)}</div><div className="history-private"><Icon name="shield" size={16} />History contains predictions and measurements only—not your original photos.</div></section>}

          <section className="settings-card"><div className="settings-icon"><Icon name="shield" size={18} /></div><div className="settings-copy"><strong>API access</strong><span>Enter a key only if your server requires one. It stays in this browser.</span></div><input id="api-key" type="password" autoComplete="off" placeholder="Optional API key" value={apiKey} onChange={(event) => setApiKey(event.target.value)} /><button onClick={saveKey}>Save key</button></section>
          <footer className="page-footer"><span>© {new Date().getFullYear()} Plant Guard AI</span><span>AI-assisted plant health · Advisory use only</span></footer>
        </div>
      </main>
    </div>
  );
}

export default App;

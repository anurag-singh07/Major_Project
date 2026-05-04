'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import DiagnosisResult from '@/components/DiagnosisResult'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://major-project-0plz.onrender.com'

const LOADING_STEPS = [
    'Preprocessing chest X-ray image…',
    'Running EfficientNet-V2-S classification…',
    'Generating Grad-CAM saliency map…',
    'Retrieving similar cases from database…',
    'Computing severity score…',
]

// Mock data for the results preview section
const MOCK_PROBS = [
    { disease: 'Pneumonia', prob: 0.72, color: '#10b981' },
    { disease: 'Infiltration', prob: 0.45, color: '#f59e0b' },
    { disease: 'Effusion', prob: 0.30, color: '#3b82f6' },
    { disease: 'Atelectasis', prob: 0.18, color: '#06b6d4' },
    { disease: 'No Finding', prob: 0.12, color: '#475569' },
]

const TRUST_SIGNALS = [
    { icon: '🔒', text: 'Data not stored permanently' },
    { icon: '🔬', text: 'For research use only' },
    { icon: '📋', text: 'NIH-validated dataset' },
    { icon: '⚕', text: 'Clinical review recommended' },
]

const FEATURES = [
    { icon: '🧠', name: 'Multi-label Detection', desc: '14 thoracic pathologies in a single forward pass' },
    { icon: '🔥', name: 'Grad-CAM Heatmaps', desc: 'Spatial attention maps over the X-ray' },
    { icon: '📊', name: 'Severity Scoring', desc: 'Quantified 0–100 score from activation area' },
    { icon: '🔍', name: 'Similar Case Retrieval', desc: 'Nearest-neighbour search on 1280-d embeddings' },
]

export default function HomePage() {
    const [file, setFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [loading, setLoading] = useState(false)
    const [stepIdx, setStepIdx] = useState(0)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    const onDrop = useCallback((accepted) => {
        if (!accepted.length) return
        const f = accepted[0]
        setFile(f); setPreview(URL.createObjectURL(f))
        setResult(null); setError(null)
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop, accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
        maxFiles: 1, maxSize: 20 * 1024 * 1024,
    })

    const analyze = async () => {
        if (!file) return
        setLoading(true); setError(null); setResult(null); setStepIdx(0)
        let si = 0
        const iv = setInterval(() => { si = (si + 1) % LOADING_STEPS.length; setStepIdx(si) }, 1800)
        try {
            const fd = new FormData(); fd.append('file', file)
            const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd })
            if (!res.ok) {
                const e = await res.json().catch(() => ({ detail: res.statusText }))
                throw new Error(e.detail || 'Analysis failed')
            }
            const data = await res.json()
            let similar = []
            try {
                const sr = await fetch(`${API_BASE}/similarity`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ embedding: data.embedding, top_k: 5 }),
                })
                if (sr.ok) { const sd = await sr.json(); similar = sd.similar_cases }
            } catch (_) { }
            setResult({ ...data, similar_cases: similar })
        } catch (e) {
            setError(e.message)
        } finally {
            clearInterval(iv); setLoading(false)
        }
    }

    const reset = () => {
        setFile(null); setPreview(null); setResult(null); setError(null)
    }

    if (result) return (
        <div className="container" style={{ paddingTop: 40, paddingBottom: 80 }}>
            <DiagnosisResult result={result} preview={preview} onReset={reset} />
        </div>
    )

    return (
        <>
            <div className="container" style={{ paddingBottom: 80 }}>

                {/* ── HERO ─────────────────────────────────── */}
                <section className="hero">
                    <div className="hero-eyebrow">
                        <span className="hero-eyebrow-dot" />
                        EfficientNet-V2-S · Grad-CAM · NIH ChestX-ray14
                    </div>

                    <h1 className="hero-title">
                        AI-Assisted Chest X-ray<br />
                        <span className="hero-title-gradient">Diagnostic Support</span>
                    </h1>

                    <p className="hero-desc">
                        Explainable multi-label classification across 14 thoracic pathologies.
                        Designed as a decision-support tool — not a replacement for clinical expertise.
                    </p>

                    {/* Stats — accurate, clinical */}
                    <div className="hero-stats">
                        {[
                            { val: '14', label: 'Pathologies' },
                            { val: 'NIH', label: 'ChestX-ray14' },
                            { val: 'Grad-CAM', label: 'Explainability' },
                            { val: 'EfficientNet', label: 'V2-S Backbone' },
                            { val: 'T4 GPU', label: 'Inference' },
                        ].map(({ val, label }) => (
                            <div key={label} className="stat-pill">
                                <span className="stat-val">{val}</span>
                                <span className="stat-label">{label}</span>
                            </div>
                        ))}
                    </div>
                </section>

                {/* ── UPLOAD + TRUST ──────────────────────── */}
                <div className="upload-section">
                    <div {...getRootProps()} className={`upload-zone${isDragActive ? ' drag-active' : ''}`}>
                        <input {...getInputProps()} />

                        {preview ? (
                            <div>
                                <div style={{ position: 'relative', display: 'inline-block', marginBottom: 16 }}>
                                    <img src={preview} alt="X-ray preview" style={{
                                        maxHeight: 220, maxWidth: '100%',
                                        borderRadius: 12, opacity: 0.92,
                                        border: '1px solid rgba(6,182,212,0.2)',
                                    }} />
                                    {isDragActive && <div className="scan-line" />}
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, marginBottom: 8 }}>
                                    <span style={{
                                        padding: '4px 12px', background: 'rgba(16,185,129,0.1)',
                                        border: '1px solid rgba(16,185,129,0.3)', borderRadius: 100,
                                        fontSize: 11, color: '#34d399', fontWeight: 600,
                                    }}>✓ Ready to Analyze</span>
                                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{file?.name}</span>
                                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                                        {(file?.size / 1024 / 1024).toFixed(1)} MB
                                    </span>
                                </div>
                                <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>Click or drop to replace image</p>
                            </div>
                        ) : (
                            <>
                                <div className="upload-icon-wrap">
                                    <div className="upload-icon-ring">🩻</div>
                                </div>
                                <div className="upload-title">
                                    {isDragActive ? 'Drop the X-ray here' : 'Upload Chest X-ray'}
                                </div>
                                <p className="upload-subtitle">
                                    Drag & drop a frontal (PA/AP) chest X-ray, or click to browse.<br />
                                    Supports PNG · JPG — max 20 MB
                                </p>
                                <div className="format-tags">
                                    {['Frontal AP', 'Lateral PA', 'PNG', 'JPG / JPEG'].map(f => (
                                        <span key={f} className="format-tag">{f}</span>
                                    ))}
                                </div>
                            </>
                        )}
                    </div>

                    {/* Analyze Button */}
                    {file && !loading && (
                        <div style={{ textAlign: 'center', marginTop: 20 }}>
                            <button onClick={analyze} className="btn btn-primary" style={{ fontSize: 15, padding: '14px 44px' }}>
                                <span>🔬</span> Run Analysis
                            </button>
                        </div>
                    )}

                    {/* Error */}
                    {error && (
                        <div style={{
                            marginTop: 16, padding: '14px 18px',
                            background: 'rgba(239,68,68,0.07)',
                            border: '1px solid rgba(239,68,68,0.25)',
                            borderRadius: 12, color: '#fca5a5', fontSize: 13,
                        }}>⚠ {error}</div>
                    )}

                    {/* Trust Signals */}
                    <div className="trust-strip">
                        {TRUST_SIGNALS.map(t => (
                            <span key={t.text} className="trust-item">
                                <span>{t.icon}</span> {t.text}
                            </span>
                        ))}
                    </div>
                </div>

                {/* ── RESULTS PREVIEW (MOCK) ────────────────── */}
                <section className="preview-section">
                    <div className="preview-label">Sample Output</div>
                    <h2 style={{ fontSize: 'clamp(22px,3vw,32px)', fontWeight: 800, marginBottom: 8, textAlign: 'center' }}>
                        What you'll see after analysis
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: 14, textAlign: 'center', marginBottom: 36, maxWidth: 500, margin: '0 auto 36px' }}>
                        Each upload produces a full diagnostic report including heatmaps, probabilities, and severity scoring.
                    </p>

                    <div className="preview-grid">
                        {/* == Heatmap Preview == */}
                        <div className="card results-grid-wide preview-card">
                            <div className="card-title"><span className="card-title-dot" />Grad-CAM Saliency Map</div>
                            <div className="heatmap-grid">
                                <div className="heatmap-img-wrap" style={{ height: 200 }}>
                                    <div style={{
                                        width: '100%', height: '100%',
                                        background: 'radial-gradient(ellipse at 45% 50%, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.03) 60%, transparent 100%)',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        flexDirection: 'column', gap: 8,
                                        borderRadius: 12,
                                        border: '1px solid rgba(255,255,255,0.05)',
                                    }}>
                                        <span style={{ fontSize: 40, opacity: 0.4 }}>🩻</span>
                                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Original X-ray</span>
                                    </div>
                                    <div className="heatmap-img-label">Original</div>
                                </div>
                                <div className="heatmap-img-wrap" style={{ height: 200 }}>
                                    <div style={{
                                        width: '100%', height: '100%',
                                        background: `
                      radial-gradient(ellipse 40% 30% at 45% 50%, rgba(239,68,68,0.6) 0%, transparent 60%),
                      radial-gradient(ellipse 60% 50% at 45% 50%, rgba(249,115,22,0.35) 0%, transparent 70%),
                      radial-gradient(ellipse 80% 70% at 45% 50%, rgba(6,182,212,0.1) 0%, transparent 80%),
                      radial-gradient(ellipse at 45% 50%, rgba(255,255,255,0.04) 0%, transparent 100%)
                    `,
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        flexDirection: 'column', gap: 8, borderRadius: 12,
                                        border: '1px solid rgba(255,100,100,0.2)',
                                    }}>
                                        <span style={{ fontSize: 40, opacity: 0.3 }}>🩻</span>
                                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Attention heatmap</span>
                                    </div>
                                    <div className="heatmap-img-label">Grad-CAM Overlay</div>
                                </div>
                            </div>
                            <div style={{ display: 'flex', gap: 16, marginTop: 14, justifyContent: 'center', flexWrap: 'wrap' }}>
                                {[
                                    { color: '#1e40af', label: 'Low attention' },
                                    { color: '#059669', label: 'Medium' },
                                    { color: '#d97706', label: 'High' },
                                    { color: '#dc2626', label: 'Critical focus' },
                                ].map(({ color, label }) => (
                                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                        <div style={{ width: 20, height: 6, borderRadius: 3, background: color, opacity: 0.85 }} />
                                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* == Severity Gauge Preview == */}
                        <div className="card preview-card">
                            <div className="card-title"><span className="card-title-dot" />Severity Score</div>
                            <div style={{ textAlign: 'center' }}>
                                <svg width="180" height="104" viewBox="0 0 180 104" style={{ display: 'block', margin: '0 auto' }}>
                                    <path d="M 18 90 A 72 72 0 0 1 162 90" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="14" strokeLinecap="round" />
                                    <path d="M 18 90 A 72 72 0 0 1 162 90" fill="none" stroke="#f97316" strokeWidth="14" strokeLinecap="round"
                                        strokeDasharray="226" strokeDashoffset="68"
                                        style={{ filter: 'drop-shadow(0 0 8px rgba(249,115,22,0.5))' }} />
                                    <text x="90" y="85" textAnchor="middle" fontSize="30" fontWeight="800" fontFamily="Space Grotesk, sans-serif" fill="#f97316">62</text>
                                    <text x="90" y="100" textAnchor="middle" fontSize="11" fill="rgba(148,163,184,0.7)" fontFamily="Inter,sans-serif" letterSpacing="1">/ 100</text>
                                </svg>
                                <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', marginTop: 4 }}>Moderate</div>
                                <div style={{ display: 'flex', gap: 6, justifyContent: 'center', marginTop: 14, flexWrap: 'wrap' }}>
                                    {[['Minimal', '#10b981'], ['Mild', '#f59e0b'], ['Moderate', '#f97316'], ['Severe', '#ef4444']].map(([l, c]) => (
                                        <span key={l} style={{
                                            padding: '3px 10px', borderRadius: 100, fontSize: 10, fontWeight: 700,
                                            background: l === 'Moderate' ? 'rgba(249,115,22,0.15)' : 'rgba(255,255,255,0.03)',
                                            color: l === 'Moderate' ? '#f97316' : 'var(--text-muted)',
                                            border: `1px solid ${l === 'Moderate' ? 'rgba(249,115,22,0.4)' : 'rgba(255,255,255,0.06)'}`,
                                        }}>{l}</span>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* == Probability Bars Preview == */}
                        <div className="card preview-card">
                            <div className="card-title"><span className="card-title-dot" />Disease Probabilities</div>
                            <div>
                                {MOCK_PROBS.map(({ disease, prob, color }) => (
                                    <div key={disease} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                                        <span style={{ width: 130, fontSize: 12, color: prob > 0.3 ? 'var(--text-primary)' : 'var(--text-muted)', fontWeight: prob > 0.3 ? 600 : 400, flexShrink: 0 }}>
                                            {disease}
                                            {prob > 0.3 && <span style={{ marginLeft: 5, fontSize: 9, padding: '1px 5px', background: `${color}25`, color, borderRadius: 100, fontWeight: 700 }}>!</span>}
                                        </span>
                                        <div style={{ flex: 1, height: 5, background: 'rgba(255,255,255,0.05)', borderRadius: 100, overflow: 'hidden' }}>
                                            <div style={{
                                                height: '100%', borderRadius: 100,
                                                width: `${prob * 100}%`,
                                                background: prob > 0.3 ? `linear-gradient(90deg, ${color}99, ${color})` : 'rgba(255,255,255,0.12)',
                                                boxShadow: prob > 0.3 ? `0 0 8px ${color}50` : 'none',
                                            }} />
                                        </div>
                                        <span style={{ width: 40, textAlign: 'right', fontSize: 12, fontWeight: 600, color: prob > 0.3 ? color : 'var(--text-muted)' }}>
                                            {(prob * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                ))}
                                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
                                    * Sample output — actual values depend on image
                                </p>
                            </div>
                        </div>

                        {/* == Similar Cases Preview == */}
                        <div className="card results-grid-wide preview-card">
                            <div className="card-title"><span className="card-title-dot" />Top-5 Similar Cases <span style={{ fontWeight: 400, color: 'var(--text-muted)', textTransform: 'none', letterSpacing: 0 }}>— retrieved via 1280-d embedding nearest-neighbour</span></div>
                            <div className="similar-grid">
                                {[
                                    { id: '00012234_034.png', match: 91.2, labels: ['Pneumonia'] },
                                    { id: '00019783_001.png', match: 88.7, labels: ['Infiltration'] },
                                    { id: '00000086_003.png', match: 85.1, labels: ['Pneumonia', 'Effusion'] },
                                    { id: '00007088_008.png', match: 82.4, labels: ['Consolidation'] },
                                    { id: '00003693_006.png', match: 79.9, labels: ['No Finding'] },
                                ].map((c, i) => (
                                    <div key={i} className="similar-case-card">
                                        <div className="similar-case-img">🩻</div>
                                        <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--cyan-light)' }}>{c.match}%</div>
                                        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>Cosine match</div>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, justifyContent: 'center', marginTop: 8 }}>
                                            {c.labels.map(l => (
                                                <span key={l} style={{ fontSize: 9, padding: '2px 6px', background: 'rgba(6,182,212,0.1)', color: 'var(--cyan)', border: '1px solid rgba(6,182,212,0.2)', borderRadius: 100, fontWeight: 600 }}>{l}</span>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </section>

                {/* ── FEATURES ──────────────────────────────── */}
                <section style={{ marginTop: 60 }}>
                    <div className="features-strip">
                        {FEATURES.map(f => (
                            <div key={f.name} className="feature-card">
                                <span className="feature-icon">{f.icon}</span>
                                <div className="feature-name">{f.name}</div>
                                <div className="feature-desc">{f.desc}</div>
                            </div>
                        ))}
                    </div>
                </section>

                {/* ── HOW IT WORKS ──────────────────────────── */}
                <section style={{ padding: '64px 0 20px', textAlign: 'center' }}>
                    <div className="section-eyebrow">METHODOLOGY</div>
                    <h2 style={{ fontSize: 'clamp(22px,3vw,32px)', fontWeight: 800, marginBottom: 40 }}>
                        How the analysis pipeline works
                    </h2>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16, maxWidth: 820, margin: '0 auto' }}>
                        {[
                            { num: '01', icon: '📤', title: 'Image Upload', desc: 'Frontal PA/AP chest X-ray is preprocessed: resized to 224×224, normalised to ImageNet statistics.' },
                            { num: '02', icon: '⚡', title: 'Model Inference', desc: 'EfficientNet-V2-S (pretrained, fine-tuned) outputs sigmoid probabilities for each of 14 pathologies.' },
                            { num: '03', icon: '📋', title: 'Explainability', desc: 'Grad-CAM generates class-specific spatial attention. Embeddings enable nearest-neighbour retrieval.' },
                        ].map(step => (
                            <div key={step.num} className="card" style={{ textAlign: 'left' }}>
                                <div style={{ fontSize: 10, fontWeight: 800, color: 'var(--cyan)', letterSpacing: '0.12em', marginBottom: 12, opacity: 0.5 }}>{step.num}</div>
                                <div style={{ fontSize: 26, marginBottom: 10 }}>{step.icon}</div>
                                <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 8, fontFamily: '"Space Grotesk",sans-serif' }}>{step.title}</div>
                                <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.7 }}>{step.desc}</div>
                            </div>
                        ))}
                    </div>
                </section>

            </div>

            {/* ── LOADING OVERLAY ───────────────────── */}
            {loading && (
                <div className="spinner-overlay">
                    <div className="spinner-ring" />
                    <div>
                        <p className="spinner-text">{LOADING_STEPS[stepIdx]}</p>
                        <div style={{ display: 'flex', gap: 6, justifyContent: 'center', marginTop: 12 }}>
                            {LOADING_STEPS.map((_, i) => (
                                <div key={i} className={`spinner-step${i === stepIdx ? ' active' : ''}`} />
                            ))}
                        </div>
                    </div>
                    <p style={{ fontSize: 11, color: 'var(--text-muted)' }}>Typically 5–15 seconds</p>
                </div>
            )}
        </>
    )
}

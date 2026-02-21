'use client'

import SeverityGauge from './SeverityGauge'
import HeatmapViewer from './HeatmapViewer'
import SimilarCases from './SimilarCases'
import ProbabilityBars from './ProbabilityBars'

const SEVERITY_COLORS = {
    Minimal: { color: '#10b981', bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.3)' },
    Mild: { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)' },
    Moderate: { color: '#f97316', bg: 'rgba(249,115,22,0.12)', border: 'rgba(249,115,22,0.3)' },
    Severe: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)', border: 'rgba(239,68,68,0.3)' },
}

export default function DiagnosisResult({ result, preview, onReset }) {
    const isNormal = result.is_normal
    const sev = result.severity
    const sevLevel = sev < 20 ? 'Minimal' : sev < 40 ? 'Mild' : sev < 65 ? 'Moderate' : 'Severe'
    const sevStyle = SEVERITY_COLORS[sevLevel]

    return (
        <div>
            {/* ── Diagnosis Banner ─────────────────────────────── */}
            <div className={`diagnosis-banner ${isNormal ? 'normal' : 'abnormal'}`}>
                <div className="diagnosis-icon">{isNormal ? '✅' : '🔍'}</div>
                <div style={{ flex: 1 }}>
                    <div className="diagnosis-title" style={{ color: isNormal ? '#34d399' : '#fca5a5' }}>
                        {isNormal ? 'No Significant Findings' : result.prediction}
                    </div>
                    <div className="diagnosis-sub">
                        {isNormal
                            ? 'X-ray appears within normal limits. Clinical correlation recommended.'
                            : `Primary finding detected. Top confidence class — clinical review advised.`}
                    </div>
                    {!isNormal && result.top_findings?.length > 0 && (
                        <div className="findings-grid">
                            {result.top_findings.map(f => (
                                <span key={f.disease} className="finding-chip" style={{
                                    color: '#fbbf24',
                                    background: 'rgba(251,191,36,0.1)',
                                    borderColor: 'rgba(251,191,36,0.3)',
                                }}>
                                    {f.disease} — {(f.probability * 100).toFixed(1)}%
                                </span>
                            ))}
                        </div>
                    )}
                </div>
                <button onClick={onReset} className="btn btn-secondary" style={{ flexShrink: 0 }}>
                    ↩ New Scan
                </button>
            </div>

            {/* ── Main Grid ─────────────────────────────────────── */}
            <div className="results-grid">

                {/* Heatmap */}
                <div className="results-grid-wide">
                    <HeatmapViewer
                        originalSrc={preview}
                        heatmapB64={result.heatmap_base64}
                        heatmapUrl={result.heatmap_url}
                        prediction={result.prediction}
                    />
                </div>

                {/* Severity */}
                <div className="card">
                    <div className="card-title"><span className="card-title-dot" />Severity Score</div>
                    <SeverityGauge score={sev} level={sevLevel} />
                    <div className="divider" />
                    <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                        <strong style={{ color: 'var(--text-primary)' }}>Severity Formula:</strong><br />
                        <code style={{ fontSize: 12, color: 'var(--cyan)' }}>
                            S = 100 × (0.7 × A<sub>fraction</sub> + 0.3 × I<sub>mean</sub>)
                        </code>
                        <br />where A = activated heatmap area, I = mean intensity
                    </div>
                </div>

                {/* Probability Bars */}
                <div className="card">
                    <div className="card-title"><span className="card-title-dot" />Disease Probabilities</div>
                    <ProbabilityBars probabilities={result.probabilities} />
                </div>

                {/* Similar Cases */}
                {result.similar_cases?.length > 0 && (
                    <div className="card results-grid-wide">
                        <div className="card-title"><span className="card-title-dot" />Top-5 Similar Cases</div>
                        <SimilarCases cases={result.similar_cases} />
                    </div>
                )}
            </div>

            {/* ── Medical Disclaimer ──────────────────────────── */}
            <div className="disclaimer">
                ⚕ <strong style={{ color: 'var(--text-secondary)' }}>Medical Disclaimer:</strong> This AI analysis is for informational
                purposes only and must not be used as a standalone diagnostic tool. Always consult a qualified radiologist.
            </div>
        </div>
    )
}

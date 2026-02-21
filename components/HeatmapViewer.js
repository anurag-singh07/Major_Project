'use client'

import { useState } from 'react'

export default function HeatmapViewer({ originalSrc, heatmapB64, heatmapUrl, prediction }) {
    const [showOverlay, setShowOverlay] = useState(true)

    const heatmapSrc = heatmapUrl
        ? heatmapUrl
        : heatmapB64
            ? `data:image/png;base64,${heatmapB64}`
            : null

    return (
        <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
                <div className="card-title" style={{ marginBottom: 0 }}>Grad-CAM Heatmap</div>
                <div style={{ display: 'flex', gap: 8 }}>
                    <button
                        onClick={() => setShowOverlay(false)}
                        className="btn btn-secondary"
                        style={{
                            padding: '6px 16px', fontSize: 12,
                            background: !showOverlay ? 'rgba(6,182,212,0.15)' : undefined,
                            borderColor: !showOverlay ? 'var(--cyan)' : undefined,
                        }}
                    >
                        Original
                    </button>
                    <button
                        onClick={() => setShowOverlay(true)}
                        className="btn btn-secondary"
                        style={{
                            padding: '6px 16px', fontSize: 12,
                            background: showOverlay ? 'rgba(6,182,212,0.15)' : undefined,
                            borderColor: showOverlay ? 'var(--cyan)' : undefined,
                        }}
                    >
                        Heatmap
                    </button>
                </div>
            </div>

            <div className="heatmap-grid">
                {/* Original X-ray */}
                <div className="heatmap-img-wrap">
                    {originalSrc ? (
                        <img
                            src={originalSrc}
                            alt="Original X-ray"
                            style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: 10 }}
                        />
                    ) : (
                        <span style={{ color: 'var(--text-muted)', fontSize: 14 }}>No image</span>
                    )}
                    <div className="heatmap-img-label">Original</div>
                </div>

                {/* Heatmap / Overlay */}
                <div className="heatmap-img-wrap">
                    {heatmapSrc ? (
                        <img
                            src={showOverlay ? heatmapSrc : originalSrc}
                            alt="Grad-CAM Heatmap"
                            style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: 10 }}
                        />
                    ) : (
                        <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, padding: 20 }}>
                            <div style={{ fontSize: 32, marginBottom: 8 }}>📡</div>
                            Heatmap not available<br />
                            <span style={{ fontSize: 11 }}>Model not loaded</span>
                        </div>
                    )}
                    <div className="heatmap-img-label">
                        {showOverlay ? `Grad-CAM: ${prediction}` : 'Original'}
                    </div>
                </div>
            </div>

            {/* Legend */}
            <div style={{
                display: 'flex', gap: 20, marginTop: 16,
                justifyContent: 'center', flexWrap: 'wrap',
            }}>
                {[
                    { color: '#0000ff', label: 'Low attention' },
                    { color: '#00ff00', label: 'Medium' },
                    { color: '#ffff00', label: 'High' },
                    { color: '#ff0000', label: 'Critical focus' },
                ].map(({ color, label }) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{
                            width: 20, height: 8, borderRadius: 4,
                            background: color, opacity: 0.8,
                        }} />
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</span>
                    </div>
                ))}
            </div>
        </div>
    )
}

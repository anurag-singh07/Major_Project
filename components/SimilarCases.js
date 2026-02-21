'use client'

import { useState } from 'react'

export default function SimilarCases({ cases }) {
    const [hovered, setHovered] = useState(null)

    if (!cases?.length) return null

    return (
        <div>
            <div className="similar-grid">
                {cases.map((c, i) => {
                    const matchPct = ((c.cosine_sim || 0) * 100).toFixed(1)
                    const diseases = Array.isArray(c.label) ? c.label.filter(l => l !== 'No Finding') : []
                    return (
                        <div
                            key={c.image_id || i}
                            className="similar-case-card"
                            onMouseEnter={() => setHovered(i)}
                            onMouseLeave={() => setHovered(null)}
                        >
                            <div className="similar-case-img">
                                🩻
                            </div>
                            <div className="similar-match">
                                {matchPct}%
                            </div>
                            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                                Cosine match
                            </div>
                            <div style={{
                                marginTop: 8, display: 'flex', flexWrap: 'wrap',
                                gap: 4, justifyContent: 'center',
                            }}>
                                {diseases.length > 0
                                    ? diseases.slice(0, 2).map(d => (
                                        <span key={d} style={{
                                            fontSize: 9, padding: '2px 6px',
                                            background: 'rgba(6,182,212,0.1)',
                                            color: 'var(--cyan)',
                                            border: '1px solid rgba(6,182,212,0.2)',
                                            borderRadius: 100, fontWeight: 600,
                                        }}>{d}</span>
                                    ))
                                    : (
                                        <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>Normal</span>
                                    )
                                }
                            </div>

                            {/* Hover tooltip */}
                            {hovered === i && (
                                <div style={{
                                    position: 'absolute', bottom: '110%', left: '50%',
                                    transform: 'translateX(-50%)',
                                    background: 'var(--bg-card)',
                                    border: '1px solid var(--border-bright)',
                                    borderRadius: 8, padding: '8px 12px',
                                    fontSize: 11, whiteSpace: 'nowrap',
                                    zIndex: 10, color: 'var(--text-secondary)',
                                    pointerEvents: 'none',
                                    boxShadow: 'var(--shadow-card)',
                                }}>
                                    <strong style={{ color: 'var(--text-primary)', display: 'block' }}>
                                        {c.image_id}
                                    </strong>
                                    <span>Cosine: {(c.cosine_sim * 100).toFixed(1)}%</span><br />
                                    <span>Euclidean: {c.euclidean_dist?.toFixed(2)}</span>
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            <p style={{
                marginTop: 16, fontSize: 12,
                color: 'var(--text-muted)', textAlign: 'center',
            }}>
                Retrieved via rank-fusion of cosine similarity + Euclidean distance (1280-d embeddings)
            </p>
        </div>
    )
}

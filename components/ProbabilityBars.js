'use client'

const DISEASE_COLORS = {
    Atelectasis: '#06b6d4',
    Cardiomegaly: '#8b5cf6',
    Effusion: '#3b82f6',
    Infiltration: '#f59e0b',
    Mass: '#ef4444',
    Nodule: '#f97316',
    Pneumonia: '#10b981',
    Pneumothorax: '#ec4899',
    Consolidation: '#6366f1',
    Edema: '#14b8a6',
    Emphysema: '#84cc16',
    Fibrosis: '#a16207',
    Pleural_Thickening: '#0ea5e9',
    Hernia: '#d97706',
}

export default function ProbabilityBars({ probabilities }) {
    if (!probabilities) return null

    const entries = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])

    return (
        <div>
            {entries.map(([disease, prob]) => {
                const pct = (prob * 100).toFixed(1)
                const color = DISEASE_COLORS[disease] || '#06b6d4'
                const isHigh = prob >= 0.3
                return (
                    <div key={disease} className="prob-row">
                        <span className="prob-label" style={{
                            color: isHigh ? 'var(--text-primary)' : 'var(--text-muted)',
                            fontWeight: isHigh ? 600 : 400,
                        }}>
                            {disease.replace(/_/g, ' ')}
                            {isHigh && (
                                <span style={{
                                    marginLeft: 6, fontSize: 10, padding: '1px 6px',
                                    background: `${color}25`, color,
                                    borderRadius: 100, fontWeight: 700,
                                }}>!</span>
                            )}
                        </span>
                        <div className="prob-bar-track">
                            <div
                                className="prob-bar-fill"
                                style={{
                                    width: `${pct}%`,
                                    background: isHigh
                                        ? `linear-gradient(90deg, ${color}99, ${color})`
                                        : 'rgba(255,255,255,0.12)',
                                    boxShadow: isHigh ? `0 0 8px ${color}50` : 'none',
                                }}
                            />
                        </div>
                        <span className="prob-value" style={{ color: isHigh ? color : 'var(--text-muted)' }}>
                            {pct}%
                        </span>
                    </div>
                )
            })}
        </div>
    )
}

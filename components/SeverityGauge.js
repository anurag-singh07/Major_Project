'use client'

const SEVERITY_COLORS = {
    Minimal: '#10b981',
    Mild: '#f59e0b',
    Moderate: '#f97316',
    Severe: '#ef4444',
}

export default function SeverityGauge({ score, level }) {
    const color = SEVERITY_COLORS[level] || '#06b6d4'
    const clamp = Math.min(Math.max(score, 0), 100)

    // SVG arc parameters
    const R = 72
    const cx = 90
    const cy = 90
    const SW = 14
    const circumference = Math.PI * R  // half-circle arc
    const strokeDashoffset = circumference - (clamp / 100) * circumference

    return (
        <div className="severity-gauge-wrap">
            <div className="severity-arc">
                <svg width="180" height="100" viewBox="0 0 180 100">
                    {/* Background arc */}
                    <path
                        d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
                        fill="none"
                        stroke="rgba(255,255,255,0.06)"
                        strokeWidth={SW}
                        strokeLinecap="round"
                    />
                    {/* Colored progress arc */}
                    <path
                        d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
                        fill="none"
                        stroke={color}
                        strokeWidth={SW}
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        strokeDashoffset={strokeDashoffset}
                        style={{
                            transition: 'stroke-dashoffset 1s cubic-bezier(0.23, 1, 0.32, 1)',
                            filter: `drop-shadow(0 0 8px ${color}60)`,
                        }}
                    />
                    {/* Score text */}
                    <text
                        x={cx} y={cy - 4}
                        textAnchor="middle"
                        fontSize="28"
                        fontWeight="800"
                        fontFamily="Space Grotesk, sans-serif"
                        fill={color}
                    >
                        {Math.round(clamp)}
                    </text>
                    <text
                        x={cx} y={cy + 14}
                        textAnchor="middle"
                        fontSize="11"
                        fill="rgba(148,163,184,0.8)"
                        fontFamily="Inter, sans-serif"
                        letterSpacing="1"
                    >
                        / 100
                    </text>
                </svg>
            </div>

            <div className="severity-label">{level}</div>

            {/* Level pills */}
            <div style={{ display: 'flex', gap: 6, justifyContent: 'center', marginTop: 14, flexWrap: 'wrap' }}>
                {['Minimal', 'Mild', 'Moderate', 'Severe'].map(l => (
                    <span key={l} style={{
                        padding: '3px 12px',
                        borderRadius: 100,
                        fontSize: 11,
                        fontWeight: 600,
                        background: l === level ? `${SEVERITY_COLORS[l]}22` : 'rgba(255,255,255,0.04)',
                        color: l === level ? SEVERITY_COLORS[l] : 'var(--text-muted)',
                        border: `1px solid ${l === level ? SEVERITY_COLORS[l] + '50' : 'rgba(255,255,255,0.08)'}`,
                        transition: 'all 0.3s',
                    }}>{l}</span>
                ))}
            </div>

            {/* Interpretation */}
            <div style={{
                marginTop: 16, padding: '10px 14px',
                background: `${color}10`,
                border: `1px solid ${color}30`,
                borderRadius: 10, fontSize: 12,
                color: 'var(--text-secondary)', lineHeight: 1.6,
                textAlign: 'left',
            }}>
                {{
                    Minimal: '✓ Very small area of concern. Likely within normal limits.',
                    Mild: '⚠ Moderate localized abnormality. Monitoring recommended.',
                    Moderate: '⚡ Significant area of abnormality. Clinical review recommended.',
                    Severe: '🚨 Large area of abnormality. Urgent clinical attention advised.',
                }[level]}
            </div>
        </div>
    )
}

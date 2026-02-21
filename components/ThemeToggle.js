'use client'

import { useEffect, useState } from 'react'

export default function ThemeToggle() {
    const [dark, setDark] = useState(true)

    useEffect(() => {
        const saved = localStorage.getItem('radsight-theme')
        const isDark = saved ? saved === 'dark' : true
        setDark(isDark)
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light')
    }, [])

    const toggle = () => {
        const next = !dark
        setDark(next)
        const theme = next ? 'dark' : 'light'
        document.documentElement.setAttribute('data-theme', theme)
        localStorage.setItem('radsight-theme', theme)
    }

    return (
        <button
            onClick={toggle}
            className="theme-toggle"
            title={dark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle theme"
        >
            <span className="theme-toggle-track">
                <span className="theme-toggle-thumb">{dark ? '🌙' : '☀️'}</span>
            </span>
            <span className="theme-toggle-label">{dark ? 'Dark' : 'Light'}</span>
        </button>
    )
}

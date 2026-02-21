import './globals.css'
import ThemeToggle from '@/components/ThemeToggle'

export const metadata = {
    title: 'Radiosight — Chest X-ray AI Diagnostic Support',
    description: 'Explainable AI chest X-ray analysis. Multi-label disease classification with Grad-CAM heatmaps, severity scoring, and similar case retrieval.',
}

export const viewport = {
    width: 'device-width', initialScale: 1,
}

export default function RootLayout({ children }) {
    return (
        <html lang="en" data-theme="dark">
            <head>
                <link rel="preconnect" href="https://fonts.googleapis.com" />
                <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
                {/* Prevent flash of wrong theme */}
                <script dangerouslySetInnerHTML={{
                    __html: `
          (function(){
            var t = localStorage.getItem('radsight-theme') || 'dark';
            document.documentElement.setAttribute('data-theme', t);
          })()
        `}} />
            </head>
            <body>
                <header className="header">
                    <div className="container">
                        <div className="header-inner">
                            <a href="/" className="logo">
                                <div className="logo-icon">🫁</div>
                                <span className="logo-text">Radiosight</span>
                            </a>

                            <div className="header-right">
                                <ThemeToggle />
                                <div className="header-dot" title="System Online" />
                            </div>
                        </div>
                    </div>
                </header>

                <main>{children}</main>

                <footer className="footer">
                    <div className="container">
                        <p>
                            Radiosight is a <strong>clinical decision support tool</strong> — not a substitute for professional medical judgment. &nbsp;·&nbsp;
                            Model trained on{' '}
                            <a href="https://nihcc.app.box.com/v/ChestXray-NIHCC" target="_blank" rel="noopener">
                                NIH ChestX-ray14
                            </a>
                            {' '}(112,120 images, 14 disease labels)
                        </p>
                    </div>
                </footer>
            </body>
        </html>
    )
}

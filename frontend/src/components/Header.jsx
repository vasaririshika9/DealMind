import React from 'react';
import { Sparkles, BrainCircuit, Activity } from 'lucide-react';

export default function Header({ isBackendConnected }) {
  return (
    <header className="border-b border-[var(--border-secondary)] bg-[var(--bg-primary)]/95 backdrop-blur-md sticky top-0 z-40 w-full">
      <div className="w-full page-container h-20 flex items-center justify-between">
        {/* Brand & Subtitle */}
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-[var(--color-cream)] to-[var(--color-green-light)] p-0.5 shadow-lg shadow-[var(--color-cream)]/10 flex items-center justify-center group hover:scale-105 transition-transform duration-300">
            <div className="w-full h-full bg-[var(--bg-primary)] rounded-[10px] flex items-center justify-center">
              <BrainCircuit className="w-6 h-6 text-[var(--color-cream)] group-hover:rotate-12 transition-transform duration-300" />
            </div>
          </div>

          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-2xl font-extrabold tracking-tight text-[var(--color-cream)]">
                DealSight AI
              </h1>
              <span className="px-2.5 py-0.5 text-[10px] font-bold tracking-wider uppercase rounded-full bg-[var(--color-cream)]/15 text-[var(--color-cream)] border border-[var(--border-primary)]">
                PRO
              </span>
            </div>
            <p className="text-xs font-medium text-[var(--color-cream)] flex items-center gap-1.5 mt-0.5">
              <Sparkles className="w-3.5 h-3.5 text-[var(--color-cream)]" />
              Your AI Sales Memory Assistant
            </p>
          </div>
        </div>

        {/* Right Status / Badge */}
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--bg-card)] border border-[var(--border-primary)] text-xs font-medium text-[var(--color-cream)]">
            <Activity className="w-3.5 h-3.5 text-[var(--color-cream)] animate-pulse" />
            <span>Groq <code className="text-[var(--color-cream)] font-mono font-bold">gpt-oss-120b</code></span>
          </div>

          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${
            isBackendConnected 
              ? 'bg-[var(--status-success)]/15 border-[var(--status-success)]/30 text-[var(--color-cream)]'
              : 'bg-[var(--status-warning)]/15 border-[var(--status-warning)]/30 text-[var(--color-cream)]'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isBackendConnected ? 'bg-[var(--status-success)] animate-ping' : 'bg-[var(--status-warning)]'}`}></span>
            {isBackendConnected ? 'Backend Live' : 'Connecting...'}
          </div>
        </div>
      </div>
    </header>
  );
}

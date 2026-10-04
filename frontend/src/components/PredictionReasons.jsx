import React from 'react';
import { CheckCircle2, AlertTriangle, Sparkles } from 'lucide-react';

export default function PredictionReasons({
  positiveSignals = [],
  warningSigns = [],
  explainerUsed = 'SHAP LinearExplainer',
}) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* WHY WE THINK THIS (Positive Signals) */}
      <div className="bg-[var(--bg-card)] border border-[var(--status-success)]/30 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between pb-3 border-b border-[var(--status-success)]/20 mb-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-[var(--status-success)]" />
              <h3 className="text-sm font-extrabold text-[var(--color-cream)]">
                WHY WE THINK THIS (Positive Signals)
              </h3>
            </div>
            <span className="text-[10px] text-[var(--status-success)] font-bold px-2 py-0.5 rounded bg-[var(--status-success)]/10">
              SHAP Attributed
            </span>
          </div>

          <ul className="space-y-2.5">
            {positiveSignals.map((sig, i) => (
              <li key={i} className="flex items-start gap-2.5 text-xs text-[var(--color-cream)] leading-relaxed">
                <span className="text-[var(--status-success)] font-black text-sm leading-none mt-0.5">✓</span>
                <span>{sig.replace(/^[✓\s]+/, '')}</span>
              </li>
            ))}
          </ul>
        </div>

        <p className="text-[11px] text-[var(--color-green-light)] mt-4 pt-3 border-t border-[var(--border-secondary)] italic">
          "Customer engagement, completed demo, and decision-maker involvement are driving progression."
        </p>
      </div>

      {/* WATCH OUT FOR (Risk Factors) */}
      <div className="bg-[var(--bg-card)] border border-[var(--status-danger)]/30 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between pb-3 border-b border-[var(--status-danger)]/20 mb-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-[var(--status-danger)]" />
              <h3 className="text-sm font-extrabold text-[var(--color-cream)]">
                WATCH OUT FOR (Key Concerns)
              </h3>
            </div>
            <span className="text-[10px] text-[var(--status-danger)] font-bold px-2 py-0.5 rounded bg-[var(--status-danger)]/10">
              Risk Factors
            </span>
          </div>

          <ul className="space-y-2.5">
            {warningSigns.map((warn, i) => (
              <li key={i} className="flex items-start gap-2.5 text-xs text-[var(--color-cream)] leading-relaxed">
                <span className="text-[var(--status-danger)] font-black text-sm leading-none mt-0.5">⚠</span>
                <span>{warn.replace(/^[⚠\s]+/, '')}</span>
              </li>
            ))}
          </ul>
        </div>

        <p className="text-[11px] text-[var(--color-green-light)] mt-4 pt-3 border-t border-[var(--border-secondary)] italic">
          "Unresolved pricing hesitation remains the primary point of friction for this deal."
        </p>
      </div>
    </div>
  );
}

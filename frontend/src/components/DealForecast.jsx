import React from 'react';
import { TrendingUp, AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function DealForecast({
  prediction = '',
  chanceMovingForward = '—',
  chanceLosing = '—',
  dealHealth = 'Needs Attention',
  riskLevel = 'Medium',
  riskScore = 50,
  likelyNextStep = 'Evaluation',
  nextStepProbability = '',
  customerInterest = 'Steady →',
}) {
  const isHealthy = dealHealth === 'Healthy Deal';
  const isAttention = dealHealth === 'Needs Attention';
  const progPct = parseInt(chanceMovingForward) || 0;

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border-primary)] rounded-2xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-[var(--border-secondary)]">
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${
            isHealthy ? 'bg-[var(--status-success)]' : isAttention ? 'bg-[var(--status-warning)]' : 'bg-[var(--status-danger)]'
          } shadow-sm`} />
          <h3 className="text-xs uppercase font-extrabold tracking-wider text-[var(--color-green-muted)]">
            Deal Forecast & Next Milestone
          </h3>
        </div>
        <span className={`text-[11px] font-extrabold px-2.5 py-0.5 rounded-full ${
          isHealthy ? 'bg-[var(--status-success)]/15 text-[var(--status-success)]' :
          isAttention ? 'bg-[var(--status-warning)]/15 text-[var(--status-warning)]' :
          'bg-[var(--status-danger)]/15 text-[var(--status-danger)]'
        }`}>
          {riskLevel} Risk ({riskScore}/100)
        </span>
      </div>

      <div className="flex flex-col sm:flex-row items-start sm:items-baseline justify-between gap-2">
        <div>
          <span className="text-3xl sm:text-4xl font-black text-[var(--color-cream)]">
            {chanceMovingForward}
          </span>
          <span className="text-xs font-bold text-[var(--status-success)] ml-2">
            chance of moving forward
          </span>
        </div>
        <div className="text-xs text-[var(--color-green-light)]">
          {chanceLosing} chance of losing
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-[var(--bg-primary)] h-2.5 rounded-full overflow-hidden border border-[var(--border-secondary)]">
        <div
          className="bg-gradient-to-r from-[var(--status-warning)] via-[var(--color-cream)] to-[var(--status-success)] h-full rounded-full transition-all duration-500"
          style={{ width: `${progPct}%` }}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 pt-2 text-xs">
        <div className="p-2.5 rounded-xl bg-[var(--bg-primary)] border border-[var(--border-secondary)]">
          <span className="text-[10px] uppercase font-bold text-[var(--color-green-muted)] block">
            Likely Next Step
          </span>
          <span className="text-sm font-extrabold text-[var(--color-cream)] mt-0.5 block">
            {likelyNextStep}
          </span>
          <span className="text-[10px] text-[var(--color-green-light)]">
            {nextStepProbability ? `${nextStepProbability} likelihood` : 'Validated transition'}
          </span>
        </div>

        <div className="p-2.5 rounded-xl bg-[var(--bg-primary)] border border-[var(--border-secondary)]">
          <span className="text-[10px] uppercase font-bold text-[var(--color-green-muted)] block">
            Customer Interest
          </span>
          <span className="text-sm font-extrabold text-[var(--color-cream)] mt-0.5 block flex items-center gap-1">
            <TrendingUp className="w-3.5 h-3.5 text-[var(--status-success)] inline" />
            {customerInterest}
          </span>
          <span className="text-[10px] text-[var(--color-green-light)]">
            Across recent interactions
          </span>
        </div>
      </div>
    </div>
  );
}

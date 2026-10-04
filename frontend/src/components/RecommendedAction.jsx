import React from 'react';
import { Zap, ArrowRight, Sparkles, MessageSquare } from 'lucide-react';

export default function RecommendedAction({
  action = 'Address the pricing concern using the ROI comparison that worked in previous discussions.',
  groqBrief = null,
  onPrepareFollowup = null,
}) {
  const displayAction = groqBrief?.what_to_do || action;

  return (
    <div className="bg-gradient-to-r from-[rgba(218,216,185,0.15)] to-[rgba(16,61,48,0.85)] border border-[var(--color-cream)] rounded-2xl p-6 shadow-xl backdrop-blur-md">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-start gap-3.5">
          <div className="p-3 rounded-2xl bg-[var(--color-cream)] text-[#072A20] shadow-md shrink-0">
            <Zap className="w-6 h-6 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-extrabold uppercase tracking-wider text-[var(--color-cream)] block">
                WHAT SHOULD I DO NEXT? (RECOMMENDED SALES ACTION)
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-[var(--color-cream)]/20 text-[var(--color-cream)] font-bold">
                Groq + Hindsight
              </span>
            </div>
            <h3 className="text-base sm:text-lg font-extrabold text-[var(--color-cream)] mt-1 leading-snug">
              "{displayAction}"
            </h3>
            <p className="text-xs text-[var(--color-green-light)] mt-1.5">
              Synthesized by Groq from SHAP deal friction factors and long-term customer memory.
            </p>
          </div>
        </div>

        <div className="shrink-0 flex items-center gap-2 w-full sm:w-auto">
          <button
            type="button"
            onClick={onPrepareFollowup || (() => alert(`Action playbook ready: ${displayAction}`))}
            className="w-full sm:w-auto px-5 py-3 rounded-xl bg-[var(--button-primary)] hover:bg-[var(--hover-primary)] text-[var(--button-primary-text)] font-extrabold text-xs shadow-lg transition-all flex items-center justify-center gap-2 cursor-pointer"
          >
            <MessageSquare className="w-4 h-4" />
            <span>Prepare Follow-up</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

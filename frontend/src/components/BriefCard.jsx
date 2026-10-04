import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  AlertTriangle, 
  Swords, 
  Smile, 
  Quote, 
  Copy, 
  Check, 
  Sparkles, 
  Building2, 
  Calendar, 
  Flag,
  ArrowRight
} from 'lucide-react';
import './BriefCard.css';

export default function BriefCard({ briefData, selectedCallNumber }) {
  const [copied, setCopied] = useState(false);

  if (!briefData) return null;

  const { customer_name, brief, history, current_call } = briefData;

  // Derive latest call metadata up to selectedCallNumber
  const activeCallRecord = current_call || (history && history.length > 0 ? history[history.length - 1] : null);
  const companyName = activeCallRecord?.company_name || 'Zenith Textiles';
  const callDate = activeCallRecord?.date || '2026-08-10';
  const dealStage = activeCallRecord?.deal_stage || 'Negotiation';
  const sentiment = activeCallRecord?.sentiment || 'Neutral';
  const competitor = activeCallRecord?.competitor_mentioned || null;
  const objection = activeCallRecord?.objection_raised || 'Price / Manager Approval';
  const keyQuote = activeCallRecord?.key_quote || null;
  const nextStep = activeCallRecord?.next_step_promised || null;

  // Sentiment badge builder using dedicated CSS classes
  const renderSentimentBadge = (s) => {
    if (!s) return null;
    const lower = s.toLowerCase();
    
    if (lower.includes('positive') || lower.includes('ready') || lower.includes('won') || lower.includes('enthusiastic')) {
      return (
        <span className="brief-sentiment-badge positive">
          <span className="brief-sentiment-dot positive" />
          Positive / Ready to Close ({s})
        </span>
      );
    }
    if (lower.includes('neutral') || lower.includes('warming')) {
      return (
        <span className="brief-sentiment-badge warming">
          <span className="brief-sentiment-dot warming" />
          Neutral / Warming Up ({s})
        </span>
      );
    }
    // Hesitant / Cautious
    return (
      <span className="brief-sentiment-badge hesitant">
        <span className="brief-sentiment-dot hesitant" />
        Hesitant / Cautious ({s})
      </span>
    );
  };

  const handleCopyBrief = () => {
    navigator.clipboard.writeText(brief);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      key={`brief-call-${selectedCallNumber}-${customer_name}`}
      initial={{ opacity: 0, y: 25, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.98 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className="brief-card-container"
    >
      <div className="brief-card">
        {/* Glowing Top Accent Border */}
        <div className="brief-card-top-gradient" />

        {/* Card Header Info */}
        <div className="brief-card-header">
          <div className="brief-card-header-left">
            <div className="brief-card-header-meta">
              <span className="brief-call-badge">
                Call #{selectedCallNumber} Briefing
              </span>
              <span className="brief-date-tag">
                <Calendar className="brief-date-icon" />
                <span>{callDate}</span>
              </span>
            </div>

            <h2 className="brief-customer-name">
              {customer_name}
            </h2>

            <div className="brief-company-info">
              <Building2 className="brief-company-icon" />
              <span className="brief-company-name">{companyName}</span>
              <span className="brief-meta-separator">•</span>
              <span className="brief-stage-tag">
                <Flag className="brief-date-icon" />
                <span>Stage: <strong>{dealStage}</strong></span>
              </span>
            </div>
          </div>

          {/* Quick Copy Button */}
          <button
            type="button"
            onClick={handleCopyBrief}
            className={`brief-copy-btn ${copied ? 'copied' : ''}`}
          >
            {copied ? (
              <>
                <Check className="brief-copy-icon copied stroke-[3]" />
                <span>Brief Copied!</span>
              </>
            ) : (
              <>
                <Copy className="brief-copy-icon" />
                <span>Copy Full Brief</span>
              </>
            )}
          </button>
        </div>

        {/* Predictive Intelligence Ribbon */}
        {briefData?.prediction && (
          <div className="p-3 my-2 rounded-xl bg-[var(--bg-primary)]/80 border border-[var(--border-primary)] flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <span className="text-xs uppercase font-extrabold tracking-wider text-[var(--color-cream)] flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-[var(--color-cream)]" />
                ML Prediction:
              </span>
              <span className={`px-2 py-0.5 rounded-md text-xs font-bold border ${
                briefData.prediction.prediction === 'Progressed'
                  ? 'bg-[var(--status-success)]/15 text-[var(--status-success)] border-[var(--status-success)]/40'
                  : 'bg-[var(--status-danger)]/15 text-[var(--status-danger)] border-[var(--status-danger)]/40'
              }`}>
                {briefData.prediction.prediction} ({Math.round((briefData.prediction.progress_probability || 0) * 100)}% Progress Prob)
              </span>
            </div>

            <div className="flex items-center gap-4 text-xs">
              <span className="text-[var(--color-cream)]">
                Risk Level: <strong style={{
                  color: briefData.prediction.risk_level === 'Low' ? 'var(--status-success)' : briefData.prediction.risk_level === 'Medium' ? 'var(--status-warning)' : 'var(--status-danger)'
                }}>{briefData.prediction.risk_level} ({briefData.prediction.risk_score}/100)</strong>
              </span>
              <span className="text-[var(--color-green-light)]">
                Next Stage: <strong className="text-[var(--color-cream)]">{briefData.prediction.predicted_next_stage}</strong>
              </span>
            </div>
          </div>
        )}

        {/* Structured Highlight Grid */}
        <div className="brief-highlight-grid">
          {/* Biggest Objection Card */}
          <div className="brief-metric-card objection-card">
            <div>
              <div className="brief-metric-header danger">
                <AlertTriangle className="brief-metric-icon" />
                <span>Biggest Objection</span>
              </div>
              <p className="brief-metric-value">
                {objection || "None identified"}
              </p>
            </div>
            <div className="brief-metric-caption">
              Primary sales hurdle raised
            </div>
          </div>

          {/* Competitor Mentioned Card */}
          <div className="brief-metric-card">
            <div>
              <div className="brief-metric-header cream">
                <Swords className="brief-metric-icon" />
                <span>Competitor Mentioned</span>
              </div>
              {competitor ? (
                <span className="brief-competitor-pill">
                  {competitor}
                </span>
              ) : (
                <span className="brief-metric-value italic">
                  No competitor cited
                </span>
              )}
            </div>
            <div className="brief-metric-caption">
              {competitor ? 'Active market rival' : 'Sole option under review'}
            </div>
          </div>

          {/* Current Sentiment Card */}
          <div className="brief-metric-card">
            <div>
              <div className="brief-metric-header cream">
                <Smile className="brief-metric-icon" />
                <span>Current Sentiment</span>
              </div>
              <div>{renderSentimentBadge(sentiment)}</div>
            </div>
            <div className="brief-metric-caption">
              Buyer mood on Call #{selectedCallNumber}
            </div>
          </div>
        </div>

        {/* AI Brief Content */}
        <div className="brief-ai-section">
          <div className="brief-ai-header">
            <Sparkles className="brief-ai-icon" />
            <span>AI Pre-Call Executive Briefing</span>
          </div>
          <div className="brief-ai-box">
            {brief}
          </div>
        </div>

        {/* Quote & Next Step Promised Footer */}
        {keyQuote && (
          <div className="brief-quote-footer">
            <Quote className="brief-quote-icon" />
            <div className="brief-quote-content">
              <p className="brief-quote-heading">
                Key Customer Quote (Call #{selectedCallNumber})
              </p>
              <p className="brief-quote-text">
                "{keyQuote}"
              </p>
              {nextStep && (
                <p className="brief-next-step">
                  <ArrowRight className="brief-next-step-icon" />
                  <span>Next Step Promised: <span className="brief-next-step-val">{nextStep}</span></span>
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}

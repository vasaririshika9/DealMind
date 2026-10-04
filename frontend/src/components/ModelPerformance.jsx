import React, { useState, useEffect } from 'react';
import {
  Award,
  BarChart3,
  CheckCircle,
  HelpCircle,
  ShieldAlert,
  Target,
  Zap,
  TrendingUp,
} from 'lucide-react';
import { API_URL } from '../services/api';
import './ModelPerformance.css';

export default function ModelPerformance({ apiUrl = API_URL }) {
  const [metricsData, setMetricsData] = useState(null);
  const [featuresData, setFeaturesData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchMLData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [mRes, fRes] = await Promise.all([
        fetch(`${apiUrl}/model-metrics`),
        fetch(`${apiUrl}/feature-importance`),
      ]);
      if (mRes.ok) {
        const mJson = await mRes.json();
        setMetricsData(mJson);
      }
      if (fRes.ok) {
        const fJson = await fRes.json();
        setFeaturesData(fJson.top_features || []);
      }
      if (!mRes.ok && !fRes.ok) {
        throw new Error('Failed to load ML metrics');
      }
    } catch (e) {
      console.error('Failed to load ML metrics:', e);
      setError('Unable to connect to DealSight server. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMLData();
  }, [apiUrl]);

  if (loading) {
    return (
      <div className="p-8 text-center text-[var(--color-cream)]">
        <Zap className="w-8 h-8 mx-auto animate-bounce mb-3 text-[var(--color-cream)]" />
        <p>Connecting to DealSight... Loading Model Evaluation & Comparison Metrics...</p>
      </div>
    );
  }

  if (error && !metricsData) {
    return (
      <div className="p-8 text-center text-[var(--color-cream)] bg-[var(--bg-card)] border border-[var(--border-primary)] rounded-2xl">
        <ShieldAlert className="w-8 h-8 mx-auto text-[var(--status-danger)] mb-3" />
        <p className="font-bold text-sm mb-4">{error}</p>
        <button
          onClick={fetchMLData}
          className="px-4 py-2 bg-[var(--color-cream)]/10 hover:bg-[var(--color-cream)]/20 border border-[var(--border-secondary)] rounded-xl text-xs font-bold transition-all text-[var(--color-cream)]"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  const bestModel = metricsData?.best_model || 'Logistic Regression';
  const bestMetrics = metricsData?.metrics || {};
  const comparison = metricsData?.all_models || [];
  const evalReport = metricsData?.holdout_evaluation || {};
  const cm = evalReport?.confusion_matrix || {};

  return (
    <div className="model-performance-wrapper space-y-8">
      {/* Header Banner */}
      <div className="best-model-banner">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[var(--status-success)]/15 border border-[var(--status-success)]/30 text-[var(--status-success)] text-xs font-bold mb-2">
              <Award className="w-4 h-4" />
              Selected Production Model
            </div>
            <h2 className="text-2xl font-extrabold text-[var(--color-cream)]">
              {bestModel} (Calibrated Pipeline)
            </h2>
            <p className="text-xs text-[var(--color-green-light)] mt-1 max-w-2xl">
              Selected using stratified validation across historical deals.
              {bestMetrics.roc_auc && (
                <> Optimized for high <strong>ROC-AUC ({(bestMetrics.roc_auc).toFixed(3)})</strong></>
              )}
              {bestMetrics.recall_lost && (
                <> and maximized <strong>Recall on Lost Deals ({(bestMetrics.recall_lost * 100).toFixed(1)}%)</strong> to stop pipeline slippage.</>
              )}
            </p>
          </div>

          <div className="flex gap-4">
            <div className="kpi-mini">
              <span className="text-[10px] uppercase font-bold text-[var(--color-green-muted)]">ROC-AUC</span>
              <span className="text-xl font-extrabold text-[var(--color-cream)]">
                {bestMetrics.roc_auc ? (bestMetrics.roc_auc * 100).toFixed(1) + '%' : '—'}
              </span>
            </div>
            <div className="kpi-mini">
              <span className="text-[10px] uppercase font-bold text-[var(--color-green-muted)]">Accuracy</span>
              <span className="text-xl font-extrabold text-[var(--color-cream)]">
                {bestMetrics.accuracy ? (bestMetrics.accuracy * 100).toFixed(1) + '%' : '—'}
              </span>
            </div>
            <div className="kpi-mini highlight">
              <span className="text-[10px] uppercase font-bold text-[var(--status-success)]">Recall (Lost)</span>
              <span className="text-xl font-extrabold text-[var(--status-success)]">
                {bestMetrics.recall_lost ? (bestMetrics.recall_lost * 100).toFixed(1) + '%' : '—'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="perf-section-card">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-[var(--color-cream)]" />
          <h3 className="text-lg font-bold text-[var(--color-cream)]">
            Candidate Model Benchmark & Comparison
          </h3>
        </div>

        <div className="overflow-x-auto">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall (Prog)</th>
                <th>Recall (Lost / Risk)</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((m, idx) => {
                const isSelected = m.model_name === bestModel;
                return (
                  <tr key={idx} className={isSelected ? 'selected-row' : ''}>
                    <td className="font-bold flex items-center gap-2">
                      {isSelected && <CheckCircle className="w-4 h-4 text-[var(--status-success)]" />}
                      <span>{m.model_name}</span>
                    </td>
                    <td>{(m.accuracy * 100).toFixed(1)}%</td>
                    <td>{(m.precision * 100).toFixed(1)}%</td>
                    <td>{(m.recall_progressed * 100).toFixed(1)}%</td>
                    <td className="font-bold text-[var(--color-cream)]">{(m.recall_lost * 100).toFixed(1)}%</td>
                    <td>{(m.f1_score * 100).toFixed(1)}%</td>
                    <td className="font-bold text-[var(--status-success)]">{(m.roc_auc * 100).toFixed(1)}%</td>
                    <td>
                      {isSelected ? (
                        <span className="badge-prod">Production</span>
                      ) : (
                        <span className="badge-benchmark">Evaluated</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Business Rationale Callout */}
        <div className="business-rationale mt-4">
          <HelpCircle className="w-5 h-5 text-[var(--color-cream)] shrink-0 mt-0.5" />
          <div className="text-xs text-[var(--color-green-light)]">
            <strong className="text-[var(--color-cream)]">Data Science & Business Rationale:</strong> In enterprise B2B sales forecasting, a false negative (predicting a high-risk deal is safe) leads to sudden quarter slippage and wasted executive attention. Therefore, our model selection criteria penalizes missed risks and optimizes for <strong>Recall on Lost deals</strong> and <strong>ROC-AUC calibration</strong> over unweighted raw accuracy.
          </div>
        </div>
      </div>

      {/* Confusion Matrix & Feature Importance Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix */}
        <div className="perf-section-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-[var(--color-cream)] flex items-center gap-2">
              <Target className="w-4 h-4 text-[var(--color-cream)]" />
              Holdout Test Set Confusion Matrix (N = 375)
            </h3>
            <span className="text-xs text-[var(--color-green-muted)]">15% Holdout</span>
          </div>

          <div className="cm-grid">
            <div className="cm-cell true-neg">
              <span className="cm-label">True Negatives (Lost)</span>
              <span className="cm-val">{cm.correctly_identified_lost_deals ?? cm.true_negatives_lost ?? '—'}</span>
              <span className="cm-sub">Correctly caught deal risks</span>
            </div>
            <div className="cm-cell false-pos">
              <span className="cm-label">False Positives (False Alarm)</span>
              <span className="cm-val">{cm.incorrectly_identified_progressing_deals ?? cm.false_positives_predicted_progressed_but_lost ?? '—'}</span>
              <span className="cm-sub">Flagged safe but slipped</span>
            </div>
            <div className="cm-cell false-neg">
              <span className="cm-label">False Negatives (Missed)</span>
              <span className="cm-val">{cm.missed_lost_deals ?? cm.false_negatives_predicted_lost_but_progressed ?? '—'}</span>
              <span className="cm-sub">Flagged risky but won</span>
            </div>
            <div className="cm-cell true-pos">
              <span className="cm-label">True Positives (Progressed)</span>
              <span className="cm-val">{cm.correctly_identified_progressing_deals ?? cm.true_positives_progressed ?? '—'}</span>
              <span className="cm-sub">Accurately predicted progression</span>
            </div>
          </div>
        </div>

        {/* Feature Importance */}
        <div className="perf-section-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-[var(--color-cream)] flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-[var(--color-cream)]" />
              Top Predictive Features (Model Weights)
            </h3>
            <span className="text-xs text-[var(--color-green-muted)]">Standardized Impact</span>
          </div>

          <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
            {featuresData.slice(0, 10).map((feat, i) => (
              <div key={i} className="text-xs">
                <div className="flex justify-between mb-1 font-medium">
                  <span className="text-[var(--color-cream)]">
                    {i + 1}. {feat.feature.replace(/_/g, ' ')}
                  </span>
                  <span className="font-bold text-[var(--color-cream)]">{feat.importance}</span>
                </div>
                <div className="w-full bg-[var(--bg-primary)] h-1.5 rounded-full overflow-hidden">
                  <div
                    className="bg-[var(--color-cream)] h-full rounded-full"
                    style={{ width: `${Math.min(100, (feat.importance / (featuresData[0]?.importance || 1.5)) * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

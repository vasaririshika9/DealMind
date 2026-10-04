
import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import CustomerSelector from './components/CustomerSelector';
import CallTimeline from './components/CallTimeline';
import BriefCard from './components/BriefCard';
import LoadingState from './components/LoadingState';
import AddCallModal from './components/AddCallModal';
import PredictiveDashboard from './components/PredictiveDashboard';
import ModelPerformance from './components/ModelPerformance';
import DatasetDashboard from './components/DatasetDashboard';
import {
  PlusCircle,
  RefreshCw,
  AlertCircle,
  Sparkles,
  BrainCircuit,
  TrendingUp,
  BarChart3,
  PhoneCall,
  Sliders,
  Database,
} from 'lucide-react';

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8001').replace(/\/+$/, '');

export default function App() {
  const [activeTab, setActiveTab] = useState('brief'); // 'brief' | 'predictive' | 'models'
  const [customers, setCustomers] = useState([]);
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [selectedCallNumber, setSelectedCallNumber] = useState(1);
  const [briefData, setBriefData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [error, setError] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Fetch customer list on mount
  useEffect(() => {
    fetchCustomers();
  }, []);

  const fetchCustomers = async () => {
    try {
      console.log('[FRONTEND] Fetching customers list from backend...');

      const response = await fetch(`${API_URL}/customers`);

      if (response.ok) {
        const data = await response.json();

        console.log('[FRONTEND] Customers loaded:', data);

        setCustomers(data);
        setIsBackendConnected(true);

        if (data.length > 0 && !selectedCustomer) {
          const rahul =
            data.find((c) => c.customer_name === 'Rahul Sharma') || data[0];

          setSelectedCustomer(rahul);
          setSelectedCallNumber(1);
        }
      } else {
        throw new Error(
          `Failed to fetch customers (HTTP ${response.status})`
        );
      }
    } catch (err) {
      console.warn(
        '[FRONTEND] Backend unavailable, using fallback mock data:',
        err
      );

      setIsBackendConnected(false);

      const fallbackList = [
        {
          customer_name: 'Rahul Sharma',
          company_name: 'Zenith Textiles',
          total_calls: 5,
          latest_sentiment: 'positive',
        },
        {
          customer_name: 'Priya Nair',
          company_name: 'Coastal Foods Ltd',
          total_calls: 3,
          latest_sentiment: 'positive',
        },
        {
          customer_name: 'Arjun Mehta',
          company_name: 'BrightPath Logistics',
          total_calls: 2,
          latest_sentiment: 'neutral',
        },
        {
          customer_name: 'Sneha Kulkarni',
          company_name: 'Everline Retail',
          total_calls: 2,
          latest_sentiment: 'hesitant',
        },
      ];

      setCustomers(fallbackList);

      if (!selectedCustomer) {
        setSelectedCustomer(fallbackList[0]);
        setSelectedCallNumber(1);
      }
    }
  };

  const customerName = selectedCustomer?.customer_name;

  // Fetch executive brief whenever customerName or selectedCallNumber changes
  useEffect(() => {
    let isCancelled = false;

    if (!customerName) return;

    const loadBrief = async () => {
      setLoading(true);
      setError(null);

      const targetUrl = `${API_URL}/brief/${encodeURIComponent(
        customerName
      )}/${selectedCallNumber}`;

      console.log(
        `[FRONTEND] Fetching brief for '${customerName}' call #${selectedCallNumber} from ${targetUrl}...`
      );

      // 25-second frontend timeout
      const controller = new AbortController();

      const timeoutId = setTimeout(() => {
        console.warn(
          `[FRONTEND TIMEOUT] 25s timeout triggered for '${customerName}' call #${selectedCallNumber}`
        );

        controller.abort();
      }, 25000);

      try {
        const response = await fetch(targetUrl, {
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorJson = await response.json().catch(() => ({}));

          const message =
            errorJson.detail ||
            errorJson.error ||
            `HTTP ${response.status}: Failed to generate brief — check backend logs`;

          console.error(
            `[FRONTEND ERROR] Server returned HTTP ${response.status}:`,
            message
          );

          throw new Error(message);
        }

        const data = await response.json();

        if (isCancelled) return;

        console.log(
          `[FRONTEND SUCCESS] Received brief data for '${customerName}' call #${selectedCallNumber}:`,
          data
        );

        setBriefData(data);
        setIsBackendConnected(true);

        // Dynamically sync total calls
        if (data.total_calls_available) {
          setSelectedCustomer((prev) => {
            if (
              !prev ||
              prev.total_calls === data.total_calls_available
            ) {
              return prev;
            }

            return {
              ...prev,
              total_calls: data.total_calls_available,
            };
          });
        }
      } catch (err) {
        clearTimeout(timeoutId);

        if (isCancelled) return;

        if (err.name === 'AbortError') {
          console.error(
            `[FRONTEND ERROR] Request timed out after 25s for '${customerName}' call #${selectedCallNumber}`
          );

          setError(
            'Failed to generate brief — request timed out after 25s. Check backend logs.'
          );
        } else {
          console.error(
            '[FRONTEND ERROR] Failed to fetch brief:',
            err
          );

          setError(
            err.message ||
            'Failed to generate brief — check backend logs'
          );
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    };

    loadBrief();

    return () => {
      isCancelled = true;
    };
  }, [customerName, selectedCallNumber]);

  const fetchBrief = async (
    cName = customerName,
    cNum = selectedCallNumber
  ) => {
    if (!cName) return;

    setLoading(true);
    setError(null);

    const targetUrl = `${API_URL}/brief/${encodeURIComponent(cName)}/${cNum}`;

    console.log(
      `[FRONTEND] Fetching brief for '${cName}' call #${cNum} from ${targetUrl}...`
    );

    const controller = new AbortController();

    const timeoutId = setTimeout(() => {
      console.warn(
        `[FRONTEND TIMEOUT] 25s timeout triggered for '${cName}' call #${cNum}`
      );

      controller.abort();
    }, 25000);

    try {
      const response = await fetch(targetUrl, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));

        const message =
          errorJson.detail ||
          errorJson.error ||
          `HTTP ${response.status}: Failed to generate brief — check backend logs`;

        console.error(
          `[FRONTEND ERROR] Server returned HTTP ${response.status}:`,
          message
        );

        throw new Error(message);
      }

      const data = await response.json();

      console.log(
        `[FRONTEND SUCCESS] Received brief data for '${cName}' call #${cNum}:`,
        data
      );

      setBriefData(data);
      setIsBackendConnected(true);

      if (data.total_calls_available) {
        setSelectedCustomer((prev) => {
          if (
            !prev ||
            prev.total_calls === data.total_calls_available
          ) {
            return prev;
          }

          return {
            ...prev,
            total_calls: data.total_calls_available,
          };
        });
      }
    } catch (err) {
      clearTimeout(timeoutId);

      if (err.name === 'AbortError') {
        console.error(
          `[FRONTEND ERROR] Request timed out after 25s for '${cName}' call #${cNum}`
        );

        setError(
          'Failed to generate brief — request timed out after 25s. Check backend logs.'
        );
      } else {
        console.error(
          '[FRONTEND ERROR] Failed to fetch brief:',
          err
        );

        setError(
          err.message ||
          'Failed to generate brief — check backend logs'
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSelectCustomer = (customer) => {
    console.log(
      '[FRONTEND] Selected customer:',
      customer.customer_name
    );

    setSelectedCustomer(customer);
    setSelectedCallNumber(1);
  };

  const handleSelectCall = (callNum) => {
    console.log(
      `[FRONTEND] Selected call number #${callNum}`
    );

    setSelectedCallNumber(callNum);
  };

  const handleCallSaved = async (customerName, newCallNum) => {
    console.log(
      `[FRONTEND] Call #${newCallNum} saved for ${customerName}. Refreshing timeline...`
    );

    await fetchCustomers();

    setSelectedCallNumber(newCallNum);

    fetchBrief(customerName, newCallNum);
  };

  const totalCallsAvailable =
    briefData?.total_calls_available ||
    selectedCustomer?.total_calls ||
    5;

  const nextCallNumber = totalCallsAvailable + 1;

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] relative flex flex-col font-sans">

      {/* Background Decorative Ambient Glow Orbs */}
      <div className="fixed top-0 left-1/4 w-96 h-96 bg-[var(--bg-secondary)]/40 rounded-full blur-[140px] pointer-events-none" />

      <div className="fixed bottom-10 right-1/4 w-96 h-96 bg-[var(--color-cream)]/5 rounded-full blur-[140px] pointer-events-none" />

      {/* Top Header */}
      <Header isBackendConnected={isBackendConnected} />

      {/* Main Content Area */}
      <main className="flex-1 w-full page-container py-8 space-y-8 z-10 dashboard-container">

        {/* Hero & App Intro */}
        <div className="text-center w-full max-w-4xl mx-auto pt-2 pb-2">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-[var(--color-cream)]/10 border border-[var(--border-primary)] text-[var(--color-cream)] text-xs font-semibold mb-3">
            <BrainCircuit className="w-4 h-4 text-[var(--color-cream)]" />
            DealSight AI • Memory + Machine Learning + Explainability
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-[var(--color-cream)]">
            AI Deal Intelligence & Predictive Analytics
          </h2>
          <p className="text-sm sm:text-base text-[var(--color-cream)] mt-2 max-w-2xl mx-auto opacity-90">
            Combine long-term customer memory with predictive machine learning to forecast deal progression, identify risk, and guide sales execution.
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex justify-center w-full pb-2">
          <div className="inline-flex flex-wrap justify-center p-1.5 rounded-2xl bg-[var(--bg-secondary)] border border-[var(--border-primary)] shadow-lg gap-1 sm:gap-2">
            <button
              type="button"
              onClick={() => setActiveTab('brief')}
              className={`px-3.5 sm:px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold flex items-center gap-2 transition-all cursor-pointer ${
                activeTab === 'brief'
                  ? 'bg-[var(--color-cream)] text-[var(--button-primary-text)] shadow-md'
                  : 'text-[var(--color-cream)] hover:text-white hover:bg-[var(--bg-card)]'
              }`}
            >
              <PhoneCall className="w-4 h-4" />
              <span>Calls & Pre-Call Brief</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('forecast')}
              className={`px-3.5 sm:px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold flex items-center gap-2 transition-all cursor-pointer ${
                activeTab === 'forecast'
                  ? 'bg-[var(--color-cream)] text-[var(--button-primary-text)] shadow-md'
                  : 'text-[var(--color-cream)] hover:text-white hover:bg-[var(--bg-card)]'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              <span>🔮 Deal Forecast</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('insights')}
              className={`px-3.5 sm:px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold flex items-center gap-2 transition-all cursor-pointer ${
                activeTab === 'insights'
                  ? 'bg-[var(--color-cream)] text-[var(--button-primary-text)] shadow-md'
                  : 'text-[var(--color-cream)] hover:text-white hover:bg-[var(--bg-card)]'
              }`}
            >
              <Database className="w-4 h-4" />
              <span>📊 Sales Insights</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('models')}
              className={`px-3.5 sm:px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold flex items-center gap-2 transition-all cursor-pointer ${
                activeTab === 'models'
                  ? 'bg-[var(--color-cream)] text-[var(--button-primary-text)] shadow-md'
                  : 'text-[var(--color-cream)] hover:text-white hover:bg-[var(--bg-card)]'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>🏆 Model Performance</span>
            </button>
          </div>
        </div>

        {/* Tab 4: Model Performance (For Judges) */}
        {activeTab === 'models' && (
          <ModelPerformance apiUrl={API_URL} />
        )}

        {/* Tab 3: Sales Insights (Dataset Dashboard) */}
        {activeTab === 'insights' && (
          <DatasetDashboard apiUrl={API_URL} />
        )}

        {/* Tab 2: Deal Forecast (The Main Prediction Screen) */}
        {activeTab === 'forecast' && (
          <div className="space-y-6">
            <CustomerSelector
              customers={customers}
              selectedCustomer={selectedCustomer}
              onSelectCustomer={handleSelectCustomer}
            />

            <PredictiveDashboard
              prediction={briefData?.prediction}
              customerName={selectedCustomer?.customer_name}
              companyName={selectedCustomer?.company_name}
              apiUrl={API_URL}
            />
          </div>
        )}

        {/* Tab 1: Call History & Executive Brief (Original Flow) */}
        {activeTab === 'brief' && (
          <>
            {/* Customer Selector */}
            <CustomerSelector
              customers={customers}
              selectedCustomer={selectedCustomer}
              onSelectCustomer={handleSelectCustomer}
            />

            {/* Action Controls & New Call Button */}
            {selectedCustomer && (
              <div className="flex flex-col sm:flex-row sm:items-center justify-between w-full border-t border-[var(--border-primary)] pt-6 gap-4">
                <div>
                  <h3 className="text-lg sm:text-xl font-bold text-[var(--color-cream)] flex items-center gap-2">
                    <span>{selectedCustomer.customer_name}</span>
                    <span className="text-xs font-normal text-[var(--color-cream)]">
                      ({selectedCustomer.company_name})
                    </span>
                  </h3>
                  <p className="text-xs text-[var(--color-cream)] font-medium">
                    Timeline records & instant pre-call briefing
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() =>
                      fetchBrief(
                        selectedCustomer.customer_name,
                        selectedCallNumber
                      )
                    }
                    className="px-3.5 py-2 rounded-xl bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] text-[var(--color-cream)] border border-[var(--border-primary)] text-xs font-semibold flex items-center gap-1.5 transition-colors"
                    title="Refresh Brief"
                  >
                    <RefreshCw
                      className={`w-3.5 h-3.5 ${
                        loading ? 'animate-spin text-[var(--color-cream)]' : ''
                      }`}
                    />
                    <span>Refresh</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setIsModalOpen(true)}
                    className="px-4 py-2 rounded-xl bg-[var(--button-primary)] hover:bg-[var(--hover-primary)] text-[var(--button-primary-text)] text-xs font-extrabold shadow-lg shadow-[var(--color-cream)]/20 flex items-center gap-1.5 transition-all transform hover:-translate-y-0.5"
                  >
                    <PlusCircle className="w-4 h-4 text-[var(--button-primary-text)]" />
                    <span>Log Call #{nextCallNumber}</span>
                  </button>
                </div>
              </div>
            )}

            {/* Call Timeline */}
            {briefData?.history && briefData.history.length > 0 && (
              <CallTimeline
                calls={briefData.history}
                selectedCallNumber={selectedCallNumber}
                onSelectCall={handleSelectCall}
              />
            )}

            {/* Loading State or Brief Card */}
            {loading ? (
              <LoadingState message="Recalling memory & calculating deal prediction..." />
            ) : error ? (
              <div className="w-full max-w-2xl mx-auto my-8 p-6 rounded-2xl bg-[var(--status-danger)]/15 border border-[var(--status-danger)]/30 text-[var(--status-danger)] text-sm text-center flex flex-col items-center gap-2">
                <AlertCircle className="w-8 h-8 text-[var(--status-danger)]" />
                <p className="font-bold">Unable to generate brief</p>
                <p className="text-xs text-[var(--status-danger)]/80">{error}</p>
                <button
                  onClick={() =>
                    fetchBrief(
                      selectedCustomer.customer_name,
                      selectedCallNumber
                    )
                  }
                  className="mt-2 px-4 py-1.5 rounded-xl bg-[var(--status-danger)]/20 hover:bg-[var(--status-danger)]/30 text-xs font-semibold text-[var(--text-primary)] border border-[var(--status-danger)]/40"
                >
                  Retry Request
                </button>
              </div>
            ) : (
              <BriefCard
                briefData={briefData}
                selectedCallNumber={selectedCallNumber}
              />
            )}
          </>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--border-secondary)] py-6 text-center text-xs text-[var(--text-muted)] mt-auto w-full page-container">

        <div className="flex items-center justify-center gap-2 mb-1">

          <Sparkles className="w-3.5 h-3.5 text-[var(--color-cream)]" />

          <span className="font-semibold text-[var(--color-cream)]">
            DealSight AI
          </span>

          — AI-powered Deal Intelligence Agent

        </div>

        <p className="text-[var(--text-muted)]">
          Built with FastAPI, Python, Groq LLM, React & Tailwind CSS
        </p>

      </footer>

      {/* Log Call Modal */}
      <AddCallModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        selectedCustomer={selectedCustomer}
        nextCallNumber={nextCallNumber}
        onCallSaved={handleCallSaved}
      />

    </div>
  );
}
import { useState, useEffect, type FormEvent } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { SetupWizard } from './components/SetupWizard';
import { Channels } from './pages/Channels';
import { ChannelDetail } from './pages/ChannelDetail';
import { ChannelVideos } from './pages/channel/ChannelVideos';
import { ChannelSpeakers } from './pages/channel/ChannelSpeakers';
import { ChannelSearch } from './pages/channel/ChannelSearch';
import { ChannelClips } from './pages/channel/ChannelClips';
import { JobQueue } from './pages/JobQueue';
import { Settings } from './pages/Settings';
import { Speakers } from './pages/Speakers';
import { SpeakerDetailPage } from './pages/SpeakerDetailPage';
import { VideoDetailPage } from './pages/video/VideoDetailPage';
import api, { API_BASE_URL, getSharePassword, getShareToken, setSharePassword } from './lib/api';

interface ExternalShareBannerStatus {
  active: boolean;
  expires_at?: string | null;
  share_url?: string | null;
  password_required?: boolean;
}

function App() {
  const [showWizard, setShowWizard] = useState(false);
  const [shareStatus, setShareStatus] = useState<ExternalShareBannerStatus | null>(null);
  const [sharePasswordPromptOpen, setSharePasswordPromptOpen] = useState(false);
  const [sharePasswordDraft, setSharePasswordDraft] = useState('');

  useEffect(() => {
    api.get('/system/setup-status')
      .then(res => {
        if (!res.data.setup_completed) {
          setShowWizard(true);
        }
      })
      .catch(() => {
        // If endpoint fails, don't block the app
      });
  }, []);

  useEffect(() => {
    const loadShareStatus = async () => {
      const shareToken = getShareToken();
      if (shareToken) {
        try {
          const res = await fetch(`${API_BASE_URL}/share/public-status?token=${encodeURIComponent(shareToken)}`);
          const data = await res.json();
          setShareStatus(data);
          setSharePasswordPromptOpen(Boolean(data?.active && data?.password_required && !getSharePassword()));
          return;
        } catch {
          setShareStatus(null);
        }
      }

      try {
        const res = await api.get('/share/status');
        setShareStatus(res.data);
      } catch {
        setShareStatus(null);
      }
    };

    void loadShareStatus();
    const interval = window.setInterval(() => {
      void loadShareStatus();
    }, 15000);
    return () => window.clearInterval(interval);
  }, []);

  const handleSharePasswordSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!sharePasswordDraft.trim()) return;
    setSharePassword(sharePasswordDraft.trim());
    window.location.reload();
  };

  return (
    <BrowserRouter>
      {showWizard && (
        <SetupWizard onClose={() => setShowWizard(false)} onComplete={() => setShowWizard(false)} />
      )}
      {sharePasswordPromptOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/65 p-4">
          <form onSubmit={handleSharePasswordSubmit} className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl">
            <div className="text-lg font-semibold text-slate-900">Shared Session Password</div>
            <p className="mt-2 text-sm text-slate-600">
              This shared Chatalogue session requires a password before the API can be used.
            </p>
            <input
              type="password"
              value={sharePasswordDraft}
              onChange={(e) => setSharePasswordDraft(e.target.value)}
              className="mt-4 w-full rounded-xl border border-slate-200 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
              placeholder="Enter share password"
              autoFocus
            />
            <div className="mt-4 flex items-center justify-end gap-2">
              <button
                type="submit"
                className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700"
              >
                Continue
              </button>
            </div>
          </form>
        </div>
      )}
      {shareStatus?.active && (
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-900">
          <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4">
            <div>
              <span className="font-semibold">External access mode is active.</span>{' '}
              {shareStatus.expires_at ? `Expires ${new Date(shareStatus.expires_at).toLocaleString()}.` : null}
            </div>
            <div className="flex items-center gap-3">
              {shareStatus.share_url && (
                <a
                  href={shareStatus.share_url}
                  target="_blank"
                  rel="noreferrer"
                  className="font-medium text-amber-800 underline"
                >
                  Open share link
                </a>
              )}
              <a href="/settings" className="font-medium text-amber-800 underline">
                Settings
              </a>
            </div>
          </div>
        </div>
      )}
      <Layout>
        <Routes>
          <Route path="/" element={<Channels />} />
          <Route path="/jobs" element={<JobQueue />} />
          <Route path="/speakers" element={<Speakers />} />
          <Route path="/speakers/:id" element={<SpeakerDetailPage />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/video/:id" element={<VideoDetailPage />} />
          <Route path="/channel/:id" element={<ChannelDetail />}>
            <Route index element={<ChannelVideos />} />
            <Route path="transcripts" element={<ChannelSearch />} />
            <Route path="speakers" element={<ChannelSpeakers />} />
            <Route path="search" element={<Navigate to="../transcripts" replace />} />
            <Route path="clips" element={<ChannelClips />} />
          </Route>
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App;

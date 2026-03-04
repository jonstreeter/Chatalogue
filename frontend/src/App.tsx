import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
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

function App() {
  return (
    <BrowserRouter>
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

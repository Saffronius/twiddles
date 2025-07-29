import { useState } from 'react';
import { ChannelTree } from './components/ChannelTree';
import { ChannelView } from './components/ChannelView';
import { SpitballModal } from './components/SpitballModal';

function App() {
  const [selectedChannelId, setSelectedChannelId] = useState<string | null>(null);
  const [showSpitball, setShowSpitball] = useState(false);

  return (
    <div className="app">
      <ChannelTree
        selectedChannelId={selectedChannelId}
        onChannelSelect={setSelectedChannelId}
      />
      <ChannelView channelId={selectedChannelId || ''} />
      
      <button
        className="spitball-button"
        onClick={() => setShowSpitball(true)}
        title="New Spitball"
      >
        ✎
      </button>

      <SpitballModal
        isOpen={showSpitball}
        onClose={() => setShowSpitball(false)}
      />
    </div>
  );
}

export default App; 
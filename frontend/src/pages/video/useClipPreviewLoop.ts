import { useEffect } from 'react';
import { useClipsStore } from '../../store/useClipsStore';
import { usePlayerStore } from '../../store/usePlayerStore';

export function useClipPreviewLoop() {
    const player = usePlayerStore((s) => s.player);
    const currentTime = usePlayerStore((s) => s.currentTime);
    const clipPreviewLoop = useClipsStore((s) => s.clipPreviewLoop);

    useEffect(() => {
        if (!clipPreviewLoop) return;
        if (currentTime >= Math.max(clipPreviewLoop.start, clipPreviewLoop.end - 0.35)) {
            try {
                if (player && typeof player.seekTo === 'function') {
                    player.seekTo(clipPreviewLoop.start, true);
                }
                if (player && typeof player.playVideo === 'function') {
                    player.playVideo();
                }
            } catch { }
        }
    }, [currentTime, clipPreviewLoop, player]);
}

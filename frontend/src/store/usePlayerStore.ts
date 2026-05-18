import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

export type UnifiedPlayer = {
    getCurrentTime?: () => number;
    seekTo?: (seconds: number, allowSeekAhead?: boolean) => void;
    playVideo?: () => void | Promise<void>;
    pauseVideo?: () => void;
    getPlaybackRate?: () => number;
    setPlaybackRate?: (rate: number) => void;
    getPlayerState?: () => number;
};

export interface PlayerState {
    player: UnifiedPlayer | null;
    currentTime: number;
    playbackRate: number;

    setPlayer: (player: UnifiedPlayer | null) => void;
    setCurrentTime: (value: number | ((currentTime: number) => number)) => void;
    setPlaybackRate: (playbackRate: number) => void;
    resetPlayerState: () => void;
}

export const usePlayerStore = create<PlayerState>()(
    devtools(
        (set) => ({
            player: null,
            currentTime: 0,
            playbackRate: 1,

            setPlayer: (player) => set({ player }, false, 'setPlayer'),
            setCurrentTime: (value) => set((state) => ({
                currentTime: typeof value === 'function' ? value(state.currentTime) : value,
            }), false, 'setCurrentTime'),
            setPlaybackRate: (playbackRate) => set({ playbackRate }, false, 'setPlaybackRate'),
            resetPlayerState: () => set({
                player: null,
                currentTime: 0,
                playbackRate: 1,
            }, false, 'resetPlayerState'),
        }),
        { name: 'PlayerStore' },
    ),
);

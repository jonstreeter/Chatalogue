import type { ReactNode } from 'react';
import { CheckCircle2, FileText } from 'lucide-react';

type Props = {
    hasTranscript: boolean;
    renderWorkbench: () => ReactNode;
    renderSnapshot: () => ReactNode;
    onNavigateToTranscript: () => void;
};

export function OptimizeTab({ hasTranscript, renderWorkbench, renderSnapshot, onNavigateToTranscript }: Props) {
    if (hasTranscript) return renderWorkbench();

    return (
        <div className="flex-1 overflow-y-auto p-6">
            <div className="mx-auto max-w-7xl space-y-6">
                {renderSnapshot()}
                <div className="rounded-3xl border border-dashed border-slate-300 bg-white px-6 py-10 text-center shadow-sm">
                    <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-slate-100 text-slate-400">
                        <CheckCircle2 size={24} />
                    </div>
                    <div className="mt-4 text-lg font-semibold text-slate-900">Transcript required</div>
                    <div className="mx-auto mt-2 max-w-2xl text-sm leading-6 text-slate-600">
                        Optimization, rollback, and benchmarking all depend on a transcript. Generate or restore the transcript first, then return to this workbench.
                    </div>
                    <button
                        type="button"
                        onClick={onNavigateToTranscript}
                        className="mt-5 inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
                    >
                        <FileText size={15} />
                        Go to Transcript
                    </button>
                </div>
            </div>
        </div>
    );
}

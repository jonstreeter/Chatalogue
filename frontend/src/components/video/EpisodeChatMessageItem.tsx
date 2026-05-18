import { Fragment, type ReactNode } from 'react';
import type { EpisodeChatCitation, EpisodeChatMessage } from '../../types';
import { User, Bot, Loader2 } from 'lucide-react';

type Props = {
  message: EpisodeChatMessage;
  onCitationClick: (citation: EpisodeChatCitation) => void;
};

const formatTime = (seconds: number) => {
  const safe = Math.max(0, Number(seconds || 0));
  const totalSeconds = Math.floor(safe);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  return `${minutes}:${String(secs).padStart(2, '0')}`;
};

const renderInline = (text: string): ReactNode[] => {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*\n]+\*)/g).filter(Boolean);
  return parts.map((part, idx) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={idx} className="font-semibold">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('*') && part.endsWith('*') && !part.startsWith('**') && part.length > 2) {
      return <em key={idx} className="italic">{part.slice(1, -1)}</em>;
    }
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return <code key={idx} className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[0.92em] text-slate-700">{part.slice(1, -1)}</code>;
    }
    return <Fragment key={idx}>{part}</Fragment>;
  });
};

const renderMarkdownish = (content: string, tone: 'user' | 'assistant') => {
  const lines = String(content || '').replace(/\r\n/g, '\n').split('\n');
  const blocks: ReactNode[] = [];
  let paragraphLines: string[] = [];
  let bulletLines: string[] = [];
  let orderedLines: string[] = [];

  const paragraphClass = tone === 'user' ? 'text-white/95' : 'text-slate-800';
  const mutedBulletClass = tone === 'user' ? 'text-white/90' : 'text-slate-700';

  const flushParagraph = () => {
    if (!paragraphLines.length) return;
    blocks.push(
      <p key={`p-${blocks.length}`} className={`text-sm leading-7 ${paragraphClass}`}>
        {renderInline(paragraphLines.join(' '))}
      </p>
    );
    paragraphLines = [];
  };

  const flushBullets = () => {
    if (!bulletLines.length) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className={`list-disc space-y-1 pl-5 text-sm leading-7 ${mutedBulletClass}`}>
        {bulletLines.map((line, idx) => <li key={idx}>{renderInline(line)}</li>)}
      </ul>
    );
    bulletLines = [];
  };

  const flushOrdered = () => {
    if (!orderedLines.length) return;
    blocks.push(
      <ol key={`ol-${blocks.length}`} className={`list-decimal space-y-1 pl-5 text-sm leading-7 ${mutedBulletClass}`}>
        {orderedLines.map((line, idx) => <li key={idx}>{renderInline(line)}</li>)}
      </ol>
    );
    orderedLines = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();
    const bulletMatch = trimmed.match(/^[-*]\s+(.+)$/);
    const orderedMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    const headingMatch = trimmed.match(/^#{1,3}\s+(.+)$/);

    if (!trimmed) {
      flushParagraph();
      flushBullets();
      flushOrdered();
      continue;
    }

    if (headingMatch) {
      flushParagraph();
      flushBullets();
      flushOrdered();
      blocks.push(
        <div key={`h-${blocks.length}`} className={`text-xs font-semibold uppercase tracking-[0.16em] ${tone === 'user' ? 'text-white/80' : 'text-slate-500'}`}>
          {renderInline(headingMatch[1])}
        </div>
      );
      continue;
    }

    if (bulletMatch) {
      flushParagraph();
      flushOrdered();
      bulletLines.push(bulletMatch[1]);
      continue;
    }

    if (orderedMatch) {
      flushParagraph();
      flushBullets();
      orderedLines.push(orderedMatch[1]);
      continue;
    }

    flushBullets();
    flushOrdered();
    paragraphLines.push(trimmed);
  }

  flushParagraph();
  flushBullets();
  flushOrdered();

  return <div className="space-y-3">{blocks}</div>;
};

const renderCitationGroup = (
  label: string,
  citations: EpisodeChatCitation[],
  startIndex: number,
  onCitationClick: (citation: EpisodeChatCitation) => void,
) => {
  if (!citations.length) return null;
  return (
    <div className="space-y-2">
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">{label}</div>
      <div className="flex flex-col gap-2">
        {citations.map((citation, idx) => {
          const isRelated = citation.citation_scope === 'related';
          return (
            <button
              key={`${citation.citation_scope}-${citation.video_id || 'current'}-${citation.chunk_id || 'no-chunk'}-${citation.start_time}-${idx}`}
              onClick={() => onCitationClick(citation)}
              className={`group flex flex-col items-start gap-1 rounded-xl border px-3 py-2 text-left transition ${
                isRelated
                  ? 'border-amber-100 bg-amber-50 hover:border-amber-200 hover:bg-amber-100/70'
                  : 'border-slate-100 bg-slate-50 hover:border-sky-200 hover:bg-sky-50'
              }`}
              title={isRelated ? 'Open source episode' : 'Jump to position'}
            >
              <div className="flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-700">
                <span className={`rounded-full px-1.5 py-0.5 text-[10px] ${
                  isRelated ? 'bg-amber-200 text-amber-800' : 'bg-slate-200 text-slate-600 group-hover:bg-sky-200 group-hover:text-sky-800'
                }`}>
                  [{startIndex + idx}]
                </span>
                {isRelated && citation.video_title ? (
                  <span className="font-semibold text-amber-700">{citation.video_title}</span>
                ) : null}
                <span>{citation.speaker_name || 'Unknown Speaker'}</span>
                <span className="font-normal text-slate-400">· {formatTime(citation.start_time)}</span>
              </div>
              <div className="line-clamp-2 text-xs leading-snug text-slate-500">
                "{citation.support_text}" {citation.segment_ids?.length ? `• ${citation.segment_ids.length} segment${citation.segment_ids.length === 1 ? '' : 's'}` : ''}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export function EpisodeChatMessageItem({ message, onCitationClick }: Props) {
  const isUser = message.role === 'user';
  const episodeCitations = message.context?.citations || [];
  const relatedCitations = message.context?.related_citations || [];
  const hasCitations = episodeCitations.length > 0 || relatedCitations.length > 0;
  const scopeLabel = message.context?.scope_mode === 'episode_related' ? 'Episode + Related' : 'Episode';

  return (
    <div className={`mb-6 flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="mr-3 flex-shrink-0">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-sky-100 text-sky-600">
            <Bot size={18} />
          </div>
        </div>
      )}

      <div className={`max-w-[85%] rounded-2xl px-5 py-4 ${isUser ? 'bg-indigo-600 text-white shadow-md' : 'border border-slate-200 bg-white text-slate-800 shadow-sm'}`}>
        {!isUser && (
          <div className="mb-2 flex items-center justify-between gap-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
            <div className="flex items-center gap-2">
              <span>AI Assistant</span>
              {message.context ? (
                <span className={`rounded-full border px-2 py-0.5 text-[10px] ${
                  message.context.scope_mode === 'episode_related'
                    ? 'border-amber-200 bg-amber-50 text-amber-700'
                    : 'border-slate-200 bg-slate-50 text-slate-500'
                }`}>
                  {scopeLabel}
                </span>
              ) : null}
            </div>
            {message.provider && message.model && (
              <span className="opacity-70">{message.provider} · {message.model}</span>
            )}
          </div>
        )}

        <div>
          {message.content ? renderMarkdownish(message.content, isUser ? 'user' : 'assistant') : null}
          {!message.content && message.status === 'running' && (
            <span className="flex items-center gap-2 text-sm italic text-slate-400">
              <Loader2 size={14} className="animate-spin" />
              Thinking...
            </span>
          )}
          {message.error && (
            <div className="mt-3 rounded-lg border border-rose-100 bg-rose-50 p-3 text-xs font-semibold text-rose-500">
              Error: {message.error}
            </div>
          )}
        </div>

        {hasCitations && (
          <div className="mt-4 border-t border-slate-100 pt-3">
            <div className="mb-2 text-[10px] font-bold uppercase tracking-[0.08em] text-slate-400">Citations</div>
            <div className="flex flex-col gap-3">
              {renderCitationGroup('Current Episode', episodeCitations, 1, onCitationClick)}
              {renderCitationGroup('Related Episodes', relatedCitations, episodeCitations.length + 1, onCitationClick)}
            </div>
          </div>
        )}
      </div>

      {isUser && (
        <div className="ml-3 flex-shrink-0">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-100 text-indigo-600">
            <User size={18} />
          </div>
        </div>
      )}
    </div>
  );
}

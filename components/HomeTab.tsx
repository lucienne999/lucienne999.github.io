import React, { useMemo } from 'react';
import { AppTab, Note, LifeMoment } from '../types';
import { siteConfig } from '../config/siteConfig';

interface HomeTabProps {
  onNavigate: (tab: AppTab, id?: string) => void;
  notes: Note[];
  moments: LifeMoment[];
}

interface TimelineItem {
  type: 'note' | 'moment';
  date: string;
  title: string;
  id: string;
  original: Note | LifeMoment;
}

const HomeTab: React.FC<HomeTabProps> = ({ onNavigate, notes, moments }) => {
  const timelineItems = useMemo(() => {
    const items: TimelineItem[] = [];

    notes.forEach(note => {
      items.push({
        type: 'note',
        date: note.createdAt,
        title: note.title,
        id: note.id,
        original: note,
      });
    });

    moments.forEach(moment => {
      const displayTitle = moment.title || moment.description;
      items.push({
        type: 'moment',
        date: moment.date,
        title: displayTitle.length > 40 ? displayTitle.slice(0, 40) + '...' : displayTitle,
        id: moment.id,
        original: moment,
      });
    });

    return items.sort((a, b) => (a.date < b.date ? 1 : -1)).slice(0, 8);
  }, [notes, moments]);

  const { home } = siteConfig;

  return (
    <div className="space-y-14 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <section className="grid min-h-[62vh] grid-cols-1 gap-5 lg:grid-cols-[1.25fr_0.75fr]">
        <div className="hard-card rounded-[2rem] p-6 sm:p-8 lg:p-10 flex flex-col justify-between overflow-hidden relative">
          <div className="absolute -right-16 -top-16 h-48 w-48 rounded-full bg-acid blur-sm" />
          <div className="relative flex flex-wrap items-center gap-3 text-xs font-display uppercase tracking-[0.2em]">
            <span className="rounded-full bg-night px-4 py-2 text-panel">{home.badge}</span>
            <span className="rounded-full border border-neutral-900 bg-panel px-4 py-2">{notes.length} Notes</span>
            <span className="rounded-full border border-neutral-900 bg-panel px-4 py-2">{moments.length} Moments</span>
          </div>

          <div className="relative mt-16 space-y-6">
            <h1 className="hero-line text-[19vw] font-extrabold uppercase sm:text-[7.8rem] lg:text-[9.5rem]">
              {home.heroTitle.map((line, index) => (
                <React.Fragment key={`${line}-${index}`}>
                  {index > 0 && <br />}
                  {line}
                </React.Fragment>
              ))}
            </h1>
            <p className="max-w-xl text-lg leading-8 text-neutral-700 md:text-xl">
              {home.heroDescription}
            </p>
          </div>
        </div>

        <div className="grid gap-5">
          <button
            onClick={() => onNavigate(AppTab.NOTES)}
            className="ink-panel rounded-[2rem] p-7 text-left card-hover group flex flex-col justify-between min-h-[17rem]"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs uppercase tracking-[0.26em] text-acid">{home.notesCard.eyebrow}</span>
              <span className="rounded-full border border-panel/30 px-3 py-1 font-mono text-xs">{notes.length}</span>
            </div>
            <div>
              <h2 className="font-display text-5xl font-extrabold uppercase tracking-[-0.06em] group-hover:text-acid transition-colors">{home.notesCard.title}</h2>
              <p className="mt-3 text-panel/70">{home.notesCard.description}</p>
            </div>
          </button>

          <button
            onClick={() => onNavigate(AppTab.LIFE)}
            className="rounded-[2rem] border-2 border-neutral-900 bg-acid p-7 text-left card-hover group flex flex-col justify-between min-h-[17rem]"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs uppercase tracking-[0.26em]">{home.lifeCard.eyebrow}</span>
              <span className="rounded-full border border-neutral-900 px-3 py-1 font-mono text-xs">{moments.length}</span>
            </div>
            <div>
              <h2 className="font-display text-5xl font-extrabold uppercase tracking-[-0.06em] group-hover:text-accent transition-colors">{home.lifeCard.title}</h2>
              <p className="mt-3 text-neutral-700">{home.lifeCard.description}</p>
            </div>
          </button>
        </div>
      </section>

        {home.tickerText.trim() && (
          <div className="ticker -mx-4 sm:-mx-6 px-4 sm:px-6 py-3 overflow-hidden">
            <div className="font-display text-sm uppercase tracking-[0.24em] whitespace-nowrap">
              {home.tickerText}
          </div>
        </div>
        )}

        <section>
        <div className="space-y-4">
          {timelineItems.map((item, index) => (
            <button
              key={`${item.type}-${item.id}`}
              onClick={() => onNavigate(item.type === 'note' ? AppTab.NOTES : AppTab.LIFE, item.id)}
              className="w-full text-left rounded-[1.5rem] border-2 border-neutral-900 bg-panel p-5 md:p-6 card-hover group"
            >
              <div className="grid gap-4 md:grid-cols-[4rem_1fr_auto] md:items-center">
                <span className="font-display text-4xl font-extrabold tracking-[-0.08em] text-neutral-300">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-3">
                    <span className={`text-[10px] px-3 py-1 rounded-full font-mono uppercase tracking-[0.2em] border border-neutral-900 ${
                      item.type === 'note' ? 'bg-acid text-neutral-900' : 'bg-night text-panel'
                    }`}>
                      {item.type === 'note' ? 'Note' : 'Life'}
                    </span>
                    <span className="text-xs text-neutral-500 font-mono">{item.date}</span>
                  </div>
                  <h3 className="text-xl md:text-2xl font-semibold text-neutral-900 group-hover:text-accent transition-colors">
                    {item.title}
                  </h3>
                </div>
                <span className="font-display text-sm uppercase tracking-[0.18em] text-neutral-500 group-hover:text-accent transition-colors">
                  Open
                </span>
              </div>
            </button>
          ))}

          {timelineItems.length === 0 && (
            <div className="text-neutral-500 italic p-4">暂无动态</div>
          )}
        </div>
      </section>
    </div>
  );
};

export default HomeTab;

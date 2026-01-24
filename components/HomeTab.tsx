import React, { useMemo } from 'react';
import { AppTab, Note, LifeMoment } from '../types';

interface HomeTabProps {
  onNavigate: (tab: AppTab) => void;
  notes: Note[];
  moments: LifeMoment[];
}

interface TimelineItem {
  type: 'note' | 'moment';
  date: string;
  title: string;
  id: string;
  cover?: string;
  original: Note | LifeMoment;
}

const HomeTab: React.FC<HomeTabProps> = ({ onNavigate, notes, moments }) => {
  const timelineItems = useMemo(() => {
    const items: TimelineItem[] = [];
    const imgRe = /!\[[^\]]*\]\(([^)]+)\)|<img[^>]*src=["']([^"']+)["'][^>]*>/i;
    
    notes.forEach(note => {
      const m = imgRe.exec(note.content);
      const cover = m ? (m[1] || m[2]) : undefined;
      items.push({
        type: 'note',
        date: note.createdAt,
        title: note.title,
        id: note.id,
        cover,
        original: note
      });
    });

    moments.forEach(moment => {
      const displayTitle = moment.title || moment.description;
      items.push({
        type: 'moment',
        date: moment.date,
        title: displayTitle.length > 30 ? displayTitle.slice(0, 30) + '...' : displayTitle,
        id: moment.id,
        cover: moment.imageUrl,
        original: moment
      });
    });

    return items.sort((a, b) => (a.date < b.date ? 1 : -1)).slice(0, 10); // Show top 10 recent items
  }, [notes, moments]);

  return (
    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header className="space-y-3">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight">404 Not Found</h1>
        <p className="text-slate-500">这个人很懒，也想写点什么</p>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <button
          onClick={() => onNavigate(AppTab.NOTES)}
          className="group p-8 bg-white rounded-3xl border border-slate-100 shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all text-left"
        >
          <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-blue-600 group-hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold mb-2">笔记</h3>
          <p className="text-slate-500">已有 {notes.length} 篇。</p>
        </button>

        <button
          onClick={() => onNavigate(AppTab.LIFE)}
          className="group p-8 bg-white rounded-3xl border border-slate-100 shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all text-left"
        >
          <div className="w-12 h-12 bg-rose-50 text-rose-600 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-rose-600 group-hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold mb-2">生活</h3>
          <p className="text-slate-500">记录了 {moments.length} 个瞬间。</p>
        </button>
      </section>

      {/* Timeline Section */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-slate-900">最新动态</h2>
        <div className="space-y-6">
          {timelineItems.map((item, index) => (
            <div 
              key={`${item.type}-${item.id}`} 
              className="bg-white rounded-2xl border border-slate-100 overflow-hidden hover:shadow-lg transition-all cursor-pointer group"
              onClick={() => onNavigate(item.type === 'note' ? AppTab.NOTES : AppTab.LIFE)}
            >
              <div className="p-5 flex flex-col justify-center">
                <div className="flex items-center gap-2 mb-2">
                  <span className={`text-[10px] px-2 py-0.5 rounded-full uppercase tracking-wider font-bold ${
                    item.type === 'note' ? 'bg-blue-50 text-blue-600' : 'bg-rose-50 text-rose-600'
                  }`}>
                    {item.type === 'note' ? '笔记' : '生活'}
                  </span>
                  <span className="text-xs text-slate-400 font-mono">{item.date}</span>
                </div>
                <h3 className="text-lg font-bold text-slate-800 leading-snug group-hover:text-indigo-600 transition-colors">
                  {item.title}
                </h3>
              </div>
            </div>
          ))}
          
          {timelineItems.length === 0 && (
            <div className="text-slate-400 italic p-4">暂无动态</div>
          )}
        </div>
      </section>
    </div>
  );
};

export default HomeTab;

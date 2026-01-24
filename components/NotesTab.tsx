
import React, { useState, useMemo } from 'react';
import { Note } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

interface NotesTabProps {
  notes: Note[];
}

const NotesTab: React.FC<NotesTabProps> = ({ notes }) => {
  const [active, setActive] = useState<Note | null>(null);

  const cards = useMemo(() => {
    const imgRe = /!\[[^\]]*\]\(([^)]+)\)|<img[^>]*src=["']([^"']+)["'][^>]*>/i;
    const stripMd = (s: string) => s
      .replace(/```[\s\S]*?```/g, '')
      .replace(/`[^`]*`/g, '')
      .replace(/\!\[[^\]]*\]\([^)]*\)/g, '')
      .replace(/\[[^\]]*\]\([^)]*\)/g, '')
      .replace(/^#{1,6}\s+/gm, '')
      .replace(/[*_~`>#-]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    return notes.map(n => {
      const m = imgRe.exec(n.content);
      const cover = m ? (m[1] || m[2]) : undefined;
      const excerpt = stripMd(n.content).slice(0, 160);
      return { note: n, cover, excerpt };
    });
  }, [notes]);

  if (active) {
    return (
      <div className="space-y-8 animate-in fade-in slide-in-from-right-4 duration-500 max-w-3xl mx-auto">
        <div className="flex items-center justify-between pb-6 border-b border-slate-100">
          <div className="space-y-1">
            <h1 className="text-3xl font-black text-slate-900">{active.title}</h1>
            <div className="text-xs text-slate-400 font-bold">{active.category} · {active.createdAt}</div>
          </div>
          <button className="text-slate-400 hover:text-slate-600" onClick={() => setActive(null)}>返回</button>
        </div>
        <div className="prose prose-slate prose-lg max-w-none prose-p:leading-6 prose-p:my-3 prose-li:leading-6 prose-li:my-1 prose-code:before:content-none prose-code:after:content-none">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm, remarkMath]} 
            rehypePlugins={[rehypeKatex]}
            components={{
              h1({node, children, ...props}) {
                return <h1 className="text-2xl font-bold mt-8 mb-4 text-slate-900" {...props}>{children}</h1>
              },
              code({node, className, children, ...props}) {
                const match = /language-(\w+)/.exec(className || '')
                const isInline = !match && !String(children).includes('\n')
                if (isInline) {
                  return (
                    <code className="bg-slate-100 text-indigo-600 px-1.5 py-0.5 rounded text-[0.9em] font-mono break-words" {...props}>
                      {children}
                    </code>
                  )
                }
                return <code className={className} {...props}>{children}</code>
              }
            }}
          >
            {active.content}
          </ReactMarkdown>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-12 animate-in fade-in slide-in-from-right-4 duration-500 max-w-6xl mx-auto">
      <div className="flex justify-between items-end pb-8 border-b border-slate-100">
        <div>
          <h1 className="text-4xl font-black text-slate-900">知识仓库</h1>
          <p className="text-slate-400 mt-2 italic">日拱一卒</p>
        </div>
      </div>

      <div className="columns-1 sm:columns-2 lg:columns-3 gap-6">
        {cards.map(({ note, cover, excerpt }) => (
          <button key={note.id} onClick={() => setActive(note)} className="break-inside-avoid w-full block bg-white rounded-2xl overflow-hidden border border-slate-100 group text-left mb-6 hover:shadow-lg transition-shadow duration-300">
            {cover && (
              <div className="overflow-hidden">
                <img src={cover} alt={note.title} className="w-full h-auto object-cover group-hover:scale-105 transition-transform duration-500" />
              </div>
            )}
            <div className="p-6 space-y-3">
              <div className="flex items-center gap-3">
                <span className="text-xs font-bold uppercase tracking-widest text-indigo-500 bg-indigo-50 px-3 py-1 rounded-lg">{note.category}</span>
                <time className="text-xs text-slate-300 font-bold">{note.createdAt}</time>
              </div>
              <h3 className="text-xl font-black text-slate-900">{note.title}</h3>
              <p className="text-slate-600 text-sm line-clamp-2">{excerpt}</p>
              <div className="flex gap-2 flex-wrap pt-2">
                {note.tags.map(tag => (
                  <span key={tag} className="text-[10px] font-bold text-slate-400">#{tag}</span>
                ))}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default NotesTab;

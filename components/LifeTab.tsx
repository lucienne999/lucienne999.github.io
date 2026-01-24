
import React, { useState } from 'react';
import { LifeMoment } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

interface LifeTabProps {
  moments: LifeMoment[];
}

const LifeTab: React.FC<LifeTabProps> = ({ moments }) => {
  const [active, setActive] = useState<LifeMoment | null>(null);

  if (active) {
    return (
      <div className="space-y-8 animate-in fade-in slide-in-from-left-4 duration-500 max-w-3xl mx-auto">
        <div className="flex items-center justify-between pb-6 border-b border-slate-100">
          <div>
            <h2 className="text-2xl font-bold text-slate-900">{active.title || '生活瞬间'}</h2>
            <div className="text-xs text-slate-400 font-bold">{active.date}{active.location ? ` · ${active.location}` : ''}</div>
          </div>
          <button className="text-slate-400 hover:text-slate-600" onClick={() => setActive(null)}>返回</button>
        </div>
        <div className="aspect-[16/9] bg-black/5 rounded-xl overflow-hidden">
          {active.imageUrl && (
            <img src={active.imageUrl} alt={active.description} className="w-full h-full object-cover" />
          )}
        </div>
        
        {/* Tags Display */}
        {active.tags && active.tags.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {active.tags.map(tag => (
              <span key={tag} className="px-3 py-1 bg-indigo-50 text-indigo-600 text-xs font-medium rounded-full">
                #{tag}
              </span>
            ))}
          </div>
        )}

        <div className="prose prose-slate prose-lg max-w-none prose-p:leading-6 prose-p:my-3 prose-li:leading-6 prose-li:my-1 prose-code:before:content-none prose-code:after:content-none">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm, remarkMath]} 
            rehypePlugins={[rehypeKatex]}
            components={{
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
            {active.content || active.description}
          </ReactMarkdown>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-left-4 duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">生活志</h1>
          <p className="text-slate-500">记录那些闪闪发光的瞬间</p>
        </div>
      </div>

      <div className="columns-1 sm:columns-2 lg:columns-3 gap-6">
        {moments.map(moment => (
          <button key={moment.id} onClick={() => setActive(moment)} className="break-inside-avoid w-full block bg-white rounded-2xl overflow-hidden border border-slate-100 group text-left mb-6 hover:shadow-lg transition-shadow duration-300">
            {moment.imageUrl && (
              <div className="overflow-hidden relative">
                <img src={moment.imageUrl} alt={moment.description} className="w-full h-auto object-cover group-hover:scale-105 transition-transform duration-500" />
                <div className="absolute top-4 right-4 bg-white/90 backdrop-blur px-3 py-1 rounded-full text-[10px] font-bold text-slate-900 uppercase shadow-sm">
                  {moment.date}
                </div>
              </div>
            )}
            <div className="p-6 space-y-3">
              {!moment.imageUrl && (
                <div className="text-[10px] font-bold text-slate-400 uppercase mb-2">
                  {moment.date}
                </div>
              )}
              {moment.title && (
                 <h3 className="text-lg font-bold text-slate-900 leading-tight">{moment.title}</h3>
              )}
              <p className="text-slate-700 font-medium leading-relaxed line-clamp-3 text-sm">{moment.description}</p>
              {moment.location && (
                <div className="flex items-center gap-1.5 text-slate-400 text-xs">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                  </svg>
                  {moment.location}
                </div>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default LifeTab;

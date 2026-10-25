
import React, { useState, useEffect, useRef } from 'react';
import { LifeMoment } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import mermaid from 'mermaid';
import { siteConfig } from '../config/siteConfig';

mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  securityLevel: 'loose',
  fontFamily: 'inherit'
});

const Mermaid = ({ chart }: { chart: string }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string>('');

  useEffect(() => {
    const renderChart = async () => {
      if (containerRef.current && chart) {
        try {
          mermaid.mermaidAPI.reset();
          const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
          const { svg } = await mermaid.render(id, chart);
          setSvg(svg);
        } catch (error) {
          console.error('Mermaid rendering error:', error);
          setSvg(`<div class="text-red-700 p-4 border-2 border-red-700 bg-red-50 rounded-2xl">图表渲染失败: ${error instanceof Error ? error.message : '未知错误'}</div>`);
        }
      }
    };
    renderChart();
  }, [chart]);

  return (
    <div 
      className="mermaid flex justify-center my-8 overflow-x-auto" 
      ref={containerRef}
      dangerouslySetInnerHTML={{ __html: svg }} 
    />
  );
};

interface LifeTabProps {
  moments: LifeMoment[];
  targetId?: string;
}

const LifeTab: React.FC<LifeTabProps> = ({ moments, targetId }) => {
  const [active, setActive] = useState<LifeMoment | null>(() => (
    targetId ? moments.find(moment => moment.id === targetId) || null : null
  ));
  const { life } = siteConfig;

  React.useEffect(() => {
    if (targetId) {
      const moment = moments.find(m => m.id === targetId);
      if (moment) setActive(moment);
    }
  }, [targetId, moments]);

  if (active) {
    return (
      <div className="space-y-8 animate-in fade-in slide-in-from-left-4 duration-500 max-w-5xl mx-auto studio-card rounded-[2rem] p-5 sm:p-8 lg:p-10">
        <div className="flex flex-col gap-5 pb-7 border-b-2 border-neutral-900 md:flex-row md:items-start md:justify-between">
          <div>
            <div className="text-xs text-neutral-500 font-mono uppercase tracking-[0.18em]">{active.date}{active.location ? ` · ${active.location}` : ''}</div>
            <h2 className="mt-2 font-display text-4xl sm:text-5xl font-extrabold tracking-[-0.06em] text-neutral-900">{active.title || '生活瞬间'}</h2>
          </div>
          <button className="rounded-full border border-neutral-900 px-4 py-2 font-display text-xs uppercase tracking-[0.18em] text-neutral-700 hover:bg-accent hover:text-panel transition-colors" onClick={() => setActive(null)}>Back</button>
        </div>
        <div className="aspect-[16/9] bg-surface rounded-[1.5rem] overflow-hidden border-2 border-neutral-900 shadow-[8px_8px_0_#17130f]">
          {active.imageUrl && (
            <img src={active.imageUrl} alt={active.description} className="w-full h-full object-cover" />
          )}
        </div>
        
        {active.tags && active.tags.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {active.tags.map(tag => (
              <span key={tag} className="px-3 py-1 bg-acid text-neutral-900 border border-neutral-900 text-xs font-mono uppercase tracking-wider rounded-full">
                #{tag}
              </span>
            ))}
          </div>
        )}

        <div className="prose prose-lg prose-site max-w-none prose-p:leading-8 prose-p:my-4 prose-li:leading-7 prose-li:my-1 prose-code:before:content-none prose-code:after:content-none prose-pre:p-0 prose-pre:m-0 prose-pre:bg-transparent">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm, remarkMath]} 
            rehypePlugins={[rehypeKatex]}
            components={{
              pre({children}) {
                return <div className="not-prose">{children}</div>
              },
              code({node, className, children, ...props}) {
                const match = /language-(\w+)/.exec(className || '')
                const isInline = !match && !String(children).includes('\n')
                if (isInline) {
                  return (
                    <code className="bg-[#dfead6] text-[#2f4a35] border border-[#b7c9ab] px-1.5 py-0.5 rounded-md text-[0.9em] font-mono break-words" {...props}>
                      {children}
                    </code>
                  )
                }
                if (match && match[1] === 'mermaid') {
                  return <Mermaid chart={String(children).replace(/\n$/, '')} />;
                }
                const { ref, ...rest } = props;
                return (
                  <div className="code-block rounded-2xl my-7 bg-[#0d0d0d] border-2 border-neutral-900 shadow-[8px_8px_0_#17130f] overflow-hidden">
                    <div className="flex items-center justify-between px-4 py-2 bg-[#17130f] border-b border-neutral-800">
                      <div className="flex gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-[#ff5f56] border border-slate-300/20" />
                        <div className="w-3 h-3 rounded-full bg-[#ffbd2e] border border-slate-300/20" />
                        <div className="w-3 h-3 rounded-full bg-[#27c93f] border border-slate-300/20" />
                      </div>
                      <span className="text-[10px] font-mono font-bold text-neutral-500 uppercase tracking-widest">
                        {match ? match[1] : 'text'}
                      </span>
                    </div>
                    <SyntaxHighlighter
                      {...rest}
                      style={oneDark}
                      language={match ? match[1] : 'python'}
                      PreTag="pre"
                      CodeTag="code"
                      className="text-sm overflow-hidden !bg-transparent !m-0"
                      showLineNumbers={true}
                      wrapLines={false}
                      customStyle={{
                        margin: 0,
                        padding: '1.5em',
                        backgroundColor: 'transparent',
                        fontSize: '0.9em',
                        lineHeight: '1.5',
                        letterSpacing: '0.05em',
                        fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",
                      }}
                      lineNumberStyle={{
                        minWidth: '2.5em',
                        paddingRight: '1em',
                        color: '#64748b',
                        textAlign: 'right',
                        userSelect: 'none',
                        backgroundColor: 'transparent',
                      }}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  </div>
                );
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
    <div className="space-y-10 animate-in fade-in slide-in-from-left-4 duration-500">
      <div className="hard-card rounded-[2rem] p-7 sm:p-9 flex flex-col gap-6 md:flex-row md:justify-between md:items-end">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.26em] text-accent mb-4">{life.eyebrow}</p>
          <h1 className="hero-line text-6xl md:text-8xl font-extrabold uppercase text-neutral-900">{life.title}</h1>
          <p className="text-neutral-600 mt-5 max-w-xl leading-7">{life.description}</p>
        </div>
        <div className="rounded-full border-2 border-neutral-900 bg-acid px-5 py-3 font-mono text-sm uppercase tracking-[0.2em]">{moments.length} {life.countLabel}</div>
      </div>

      <div className="columns-1 sm:columns-2 lg:columns-3 gap-6">
        {moments.map(moment => (
          <button key={moment.id} onClick={() => setActive(moment)} className="break-inside-avoid w-full block bg-panel rounded-[2rem] overflow-hidden border-2 border-neutral-900 group text-left mb-6 card-hover">
            {moment.imageUrl && (
              <div className="overflow-hidden relative border-b-2 border-neutral-900">
                <img src={moment.imageUrl} alt={moment.description} className="w-full h-auto object-cover group-hover:scale-105 transition-transform duration-500" />
                <div className="absolute top-4 right-4 bg-acid backdrop-blur px-3 py-1 rounded-full text-[10px] font-mono font-semibold text-neutral-900 uppercase tracking-wider border border-neutral-900">
                  {moment.date}
                </div>
              </div>
            )}
            <div className="p-6 space-y-3">
              {!moment.imageUrl && (
                <div className="text-[10px] font-mono font-semibold text-neutral-500 uppercase tracking-wider mb-2">
                  {moment.date}
                </div>
              )}
              {moment.title && (
                 <h3 className="font-display text-2xl font-extrabold tracking-[-0.05em] text-neutral-900 leading-tight group-hover:text-accent transition-colors">{moment.title}</h3>
              )}
              <p className="text-neutral-600 font-medium leading-relaxed line-clamp-3 text-sm">{moment.description}</p>
              {moment.location && (
                <div className="flex items-center gap-1.5 text-neutral-500 text-xs">
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

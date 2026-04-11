
import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Note } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import GithubSlugger from 'github-slugger';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import mermaid from 'mermaid';
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

// Initialize mermaid
mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  securityLevel: 'loose',
  fontFamily: 'inherit'
});

// Mermaid component to render the diagram
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
          setSvg(`<div class="text-red-500 p-4 border border-red-200 bg-red-50 rounded">图表渲染失败: ${error instanceof Error ? error.message : '未知错误'}</div>`);
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

const ZoomableImage = ({ src, alt, ...props }: React.ImgHTMLAttributes<HTMLImageElement>) => {
  const [isFullscreen, setIsFullscreen] = useState(false);

  return (
    <>
      <img
        src={src}
        alt={alt}
        {...props}
        onClick={() => setIsFullscreen(true)}
        className={`rounded-lg shadow-sm cursor-zoom-in hover:shadow-md transition-all duration-300 ${props.className || ''}`}
      />
      {isFullscreen && (
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/80 backdrop-blur-sm p-4 sm:p-8 animate-in fade-in duration-200 cursor-zoom-out"
          onClick={() => setIsFullscreen(false)}
        >
          <div 
            className="relative w-[95vw] h-[95vh] bg-white rounded-2xl shadow-2xl overflow-hidden cursor-default flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="absolute top-4 right-4 z-10 flex gap-2">
              <button 
                onClick={() => setIsFullscreen(false)}
                className="p-2 bg-slate-100/80 backdrop-blur text-slate-500 hover:text-slate-900 hover:bg-slate-200 rounded-full transition-colors"
                title="关闭全屏"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>
            
            <div className="flex-1 w-full h-full bg-slate-50/50">
              <TransformWrapper
                initialScale={1}
                minScale={0.1}
                maxScale={8}
                centerOnInit={true}
                wheel={{ step: 0.1 }}
              >
                {({ zoomIn, zoomOut, resetTransform }) => (
                  <>
                    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-white/90 backdrop-blur-sm p-2 rounded-xl shadow-lg border border-slate-200">
                      <button onClick={() => zoomOut()} className="p-2 hover:bg-slate-100 rounded-lg text-slate-600 transition-colors" title="缩小">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                      </button>
                      <button onClick={() => resetTransform()} className="px-3 py-1 text-sm font-medium hover:bg-slate-100 rounded-lg text-slate-600 transition-colors" title="重置">
                        重置
                      </button>
                      <button onClick={() => zoomIn()} className="p-2 hover:bg-slate-100 rounded-lg text-slate-600 transition-colors" title="放大">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                      </button>
                    </div>
                    <TransformComponent wrapperClass="!w-full !h-full" contentClass="!w-full !h-full flex items-center justify-center">
                      <div className="w-full h-full flex items-center justify-center p-8">
                        <img src={src} alt={alt} className="max-w-full max-h-full object-contain" />
                      </div>
                    </TransformComponent>
                  </>
                )}
              </TransformWrapper>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

interface NotesTabProps {
  notes: Note[];
  targetId?: string;
}

// Add Mermaid component
const MermaidRenderer = ({ chart }: { chart: string }) => {
  const [svg, setSvg] = useState<string>('');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const id = useMemo(() => `mermaid-${Math.random().toString(36).substr(2, 9)}`, []);

  React.useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
      fontFamily: 'inherit',
    });

    const renderChart = async () => {
      try {
        const { svg: svgCode } = await mermaid.render(id, chart);
        setSvg(svgCode);
      } catch (error) {
        console.error('Mermaid rendering failed:', error);
        // Fallback: render the raw code if mermaid fails
        setSvg(`<div class="text-red-500 text-sm border border-red-200 p-4 rounded bg-red-50 whitespace-pre-wrap font-mono">${chart}</div>`);
      }
    };

    if (chart) {
        renderChart();
    }
  }, [chart, id]);

  return (
    <>
      <div className="relative group my-8">
        <div 
          className="flex justify-center overflow-x-auto bg-white p-4 rounded-xl border border-slate-100 shadow-sm transition-all duration-300 hover:shadow-md cursor-zoom-in"
          onClick={() => setIsFullscreen(true)}
          dangerouslySetInnerHTML={{ __html: svg }} 
        />
        <button 
          onClick={() => setIsFullscreen(true)}
          className="absolute top-2 right-2 p-1.5 bg-white/80 backdrop-blur border border-slate-200 rounded-lg text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity hover:text-indigo-600 hover:bg-white shadow-sm"
          title="全屏查看"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 3 21 3 21 9"></polyline>
            <polyline points="9 21 3 21 3 15"></polyline>
            <line x1="21" y1="3" x2="14" y2="10"></line>
            <line x1="3" y1="21" x2="10" y2="14"></line>
          </svg>
        </button>
      </div>

      {isFullscreen && (
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/80 backdrop-blur-sm p-4 sm:p-8 animate-in fade-in duration-200 cursor-zoom-out"
          onClick={() => setIsFullscreen(false)}
        >
          <div 
            className="relative w-[95vw] h-[95vh] bg-white rounded-2xl shadow-2xl overflow-hidden cursor-default flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="absolute top-4 right-4 z-10 flex gap-2">
              <button 
                onClick={() => setIsFullscreen(false)}
                className="p-2 bg-slate-100/80 backdrop-blur text-slate-500 hover:text-slate-900 hover:bg-slate-200 rounded-full transition-colors"
                title="关闭全屏"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>
            
            <div className="flex-1 w-full h-full bg-slate-50/50">
              <TransformWrapper
                initialScale={1}
                minScale={0.1}
                maxScale={8}
                centerOnInit={true}
                wheel={{ step: 0.1 }}
              >
                {({ zoomIn, zoomOut, resetTransform }) => (
                  <>
                    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-white/90 backdrop-blur-sm p-2 rounded-xl shadow-lg border border-slate-200">
                      <button onClick={() => zoomOut()} className="p-2 hover:bg-slate-100 rounded-lg text-slate-600 transition-colors" title="缩小">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                      </button>
                      <button onClick={() => resetTransform()} className="px-3 py-1 text-sm font-medium hover:bg-slate-100 rounded-lg text-slate-600 transition-colors" title="重置">
                        重置
                      </button>
                      <button onClick={() => zoomIn()} className="p-2 hover:bg-slate-100 rounded-lg text-slate-600 transition-colors" title="放大">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                      </button>
                    </div>
                    <TransformComponent wrapperClass="!w-full !h-full" contentClass="!w-full !h-full flex items-center justify-center">
                      <div 
                        className="w-full h-full flex items-center justify-center p-8 [&>svg]:max-w-none [&>svg]:max-h-none"
                        dangerouslySetInnerHTML={{ __html: svg }} 
                      />
                    </TransformComponent>
                  </>
                )}
              </TransformWrapper>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

const NotesTab: React.FC<NotesTabProps> = ({ notes, targetId }) => {
  const [active, setActive] = useState<Note | null>(null);
  const [showToc, setShowToc] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);

  React.useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 300) {
        setShowScrollTop(true);
      } else {
        setShowScrollTop(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  React.useEffect(() => {
    if (targetId) {
      const note = notes.find(n => n.id === targetId);
      if (note) setActive(note);
    }
  }, [targetId, notes]);

  // Reset TOC visibility when switching notes
  React.useEffect(() => {
    setShowToc(true);
  }, [active]);

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

  const headings = useMemo(() => {
    if (!active) return [];
    const slugger = new GithubSlugger();
    const lines = active.content.split('\n');
    const headings = [];
    let inCodeBlock = false;

    for (const line of lines) {
      if (line.trim().startsWith('```')) {
        inCodeBlock = !inCodeBlock;
        continue;
      }
      if (inCodeBlock) continue;

      const match = line.match(/^(#{1,6})\s+(.+)$/);
      if (match) {
        headings.push({
          level: match[1].length,
          text: match[2].trim(),
          id: slugger.slug(match[2].trim())
        });
      }
    }
    return headings;
  }, [active]);

  if (active) {
    return (
      <div className="animate-in fade-in slide-in-from-right-4 duration-500 max-w-[90rem] mx-auto px-4 sm:px-6 lg:px-8 relative">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between pb-6 border-b border-slate-100">
            <div className="space-y-1">
              <h1 className="text-3xl font-black text-slate-900">{active.title}</h1>
              <div className="text-xs text-slate-400 font-bold">{active.category} · {active.createdAt}</div>
            </div>
            <div className="flex items-center gap-4">
              <button 
                onClick={() => setShowToc(!showToc)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors ${showToc ? 'bg-indigo-50 text-indigo-600' : 'text-slate-400 hover:text-slate-600 hover:bg-slate-50'}`}
                title="目录"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="3" y1="12" x2="21" y2="12"></line>
                  <line x1="3" y1="6" x2="21" y2="6"></line>
                  <line x1="3" y1="18" x2="21" y2="18"></line>
                </svg>
                <span className="text-xs font-bold uppercase tracking-wider">目录</span>
              </button>
              <button className="text-slate-400 hover:text-slate-600" onClick={() => setActive(null)}>返回</button>
            </div>
          </div>
          <div className="prose prose-slate prose-lg max-w-none prose-p:leading-6 prose-p:my-3 prose-li:leading-6 prose-li:my-1 prose-code:before:content-none prose-code:after:content-none prose-a:text-indigo-600 prose-a:no-underline hover:prose-a:underline">
            <ReactMarkdown 
              remarkPlugins={[remarkGfm, remarkMath]} 
              rehypePlugins={[rehypeKatex, rehypeSlug, [rehypeAutolinkHeadings, { behavior: 'wrap' }]]}
              components={{
              h1({node, children, ...props}) {
                return <h1 className="text-2xl font-bold mt-8 mb-4 text-slate-900" {...props}>{children}</h1>
              },
              img({node, src, alt, ...props}) {
                return <ZoomableImage src={src} alt={alt} {...props} />;
              },
              pre({children}) {
                return <div className="not-prose">{children}</div>
              },
              code({node, className, children, ...props}) {
                const match = /language-(\w+)/.exec(className || '')
                const isInline = !match && !String(children).includes('\n')
                if (isInline) {
                  return (
                    <code className="bg-slate-100 text-indigo-600 px-1.5 py-0.5 rounded text-[0.8em] font-mono break-words" {...props}>
                      {children}
                    </code>
                  )
                }
                
                // Handle Mermaid charts
                if (match && match[1] === 'mermaid') {
                  return <MermaidRenderer chart={String(children).replace(/\n$/, '')} />;
                }

                const { ref, ...rest } = props;
                return (
                  <div className="rounded-xl my-6 bg-[#fafafa] border border-slate-200 shadow-sm overflow-hidden">
                    <div className="flex items-center justify-between px-4 py-2 bg-[#f4f4f5] border-b border-slate-200">
                      <div className="flex gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-[#ff5f56] border border-slate-300/20" />
                        <div className="w-3 h-3 rounded-full bg-[#ffbd2e] border border-slate-300/20" />
                        <div className="w-3 h-3 rounded-full bg-[#27c93f] border border-slate-300/20" />
                      </div>
                      <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-widest">
                        {match ? match[1] : 'text'}
                      </span>
                    </div>
                    <SyntaxHighlighter
                      {...rest}
                      style={oneLight}
                      language={match ? match[1] : 'python'}
                      PreTag="div"
                      className="text-sm overflow-hidden !bg-transparent"
                      showLineNumbers={true}
                      customStyle={{
                        margin: 0,
                        padding: '1.5em',
                        backgroundColor: 'transparent',
                        fontSize: '0.7em',
                        lineHeight: '1.5',
                        letterSpacing: '0.05em',
                        fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",
                      }}
                      lineNumberStyle={{
                        minWidth: '2.5em',
                        paddingRight: '1em',
                        color: '#cbd5e1',
                        textAlign: 'right',
                        userSelect: 'none',
                      }}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  </div>
                );
              }
            }}
          >
            {active.content}
          </ReactMarkdown>
        </div>
        </div>

        {/* Floating TOC Panel */}
        {showToc && (
          <div className="fixed top-24 right-8 w-72 bg-white rounded-xl shadow-xl border border-slate-100 p-6 z-50 animate-in fade-in slide-in-from-right-4 duration-200">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider">目录</h4>
              <button 
                onClick={() => setShowToc(false)}
                className="text-slate-400 hover:text-slate-600"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>
            <nav className="space-y-2 max-h-[60vh] overflow-y-auto pr-2">
              {headings.length === 0 ? (
                <p className="text-sm text-slate-400 italic">暂无目录</p>
              ) : (
                headings.map((heading) => (
                  <a
                    key={heading.id}
                    href={`#${heading.id}`}
                    className={`block text-sm transition-colors duration-200 hover:text-indigo-600 ${
                      heading.level === 1 ? 'font-bold text-slate-900' :
                      heading.level === 2 ? 'pl-0 text-slate-600' :
                      heading.level === 3 ? 'pl-4 text-slate-500' :
                      'pl-8 text-slate-400'
                    }`}
                    style={{
                      paddingLeft: heading.level > 1 ? `${(heading.level - 1) * 0.75}rem` : 0
                    }}
                    onClick={(e) => {
                      e.preventDefault();
                      document.getElementById(heading.id)?.scrollIntoView({ behavior: 'smooth' });
                      // Optional: close TOC on mobile after click, keep open on desktop
                      if (window.innerWidth < 1024) setShowToc(false);
                    }}
                  >
                    {heading.text}
                  </a>
                ))
              )}
            </nav>
          </div>
        )}

        {/* Scroll to Top Button */}
        {showScrollTop && (
          <button
            onClick={scrollToTop}
            className="fixed bottom-8 right-8 p-3 bg-white rounded-full shadow-lg border border-slate-100 text-slate-400 hover:text-indigo-600 hover:scale-110 transition-all duration-300 z-40 animate-in fade-in zoom-in duration-300"
            title="回到顶部"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5"></line>
              <polyline points="5 12 12 5 19 12"></polyline>
            </svg>
          </button>
        )}
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


import React, { useState, useMemo, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Note } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import rehypeRawMath from '../lib/rehypeRawMath';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import GithubSlugger from 'github-slugger';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import mermaid from 'mermaid';
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { siteConfig } from '../config/siteConfig';

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

const resolveMarkdownAssetPath = (src?: string) => {
  if (!src) return src;
  if (/^(https?:)?\/\//.test(src) || src.startsWith('data:') || src.startsWith('#')) {
    return src;
  }
  const publicImagesMatch = src.match(/^(?:\.\.\/)*public\/images\/(.+)$/);
  if (publicImagesMatch) {
    return `${import.meta.env.BASE_URL}images/${publicImagesMatch[1]}`;
  }
  if (src.startsWith('/public/images/')) {
    return `${import.meta.env.BASE_URL}${src.slice('/public/'.length)}`;
  }
  if (src.startsWith('/images/')) {
    return `${import.meta.env.BASE_URL}${src.slice(1)}`;
  }
  return src;
};

const ZoomableImage = ({ src, alt, ...props }: React.ImgHTMLAttributes<HTMLImageElement>) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const resolvedSrc = resolveMarkdownAssetPath(src);
  const closeFullscreen = (event?: React.MouseEvent | React.PointerEvent) => {
    event?.preventDefault();
    event?.stopPropagation();
    setIsFullscreen(false);
  };

  return (
    <>
      <img
        src={resolvedSrc}
        alt={alt}
        {...props}
        onClick={() => setIsFullscreen(true)}
        className={`rounded-2xl border-2 border-neutral-900 shadow-[6px_6px_0_#17130f] cursor-zoom-in hover:-translate-y-1 hover:shadow-[8px_8px_0_#17130f] transition-all duration-300 ${props.className || ''}`}
      />
      {isFullscreen && createPortal((
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 backdrop-blur-sm p-4 sm:p-8 animate-in fade-in duration-200 cursor-zoom-out"
          onClick={() => setIsFullscreen(false)}
        >
          <div 
            className="relative w-[95vw] h-[95vh] bg-panel rounded-[2rem] shadow-2xl overflow-hidden cursor-default flex flex-col border-2 border-neutral-900"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="absolute top-4 right-4 z-[130] flex gap-2 pointer-events-auto">
              <button 
                type="button"
                onPointerDown={closeFullscreen}
                onMouseDown={(event) => event.stopPropagation()}
                onClick={closeFullscreen}
                className="p-2 bg-acid backdrop-blur text-neutral-900 hover:bg-accent hover:text-panel rounded-full border border-neutral-900 transition-colors"
                title="关闭全屏"
                aria-label="关闭全屏"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>
            
            <div className="flex-1 w-full h-full bg-surface">
              <TransformWrapper
                initialScale={1}
                minScale={0.1}
                maxScale={8}
                centerOnInit={true}
                wheel={{ step: 0.1 }}
              >
                {({ zoomIn, zoomOut, resetTransform }) => (
                  <>
                    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-panel/90 backdrop-blur-sm p-2 rounded-2xl shadow-lg border-2 border-neutral-900">
                      <button onClick={() => zoomOut()} className="p-2 hover:bg-acid rounded-xl text-neutral-700 transition-colors" title="缩小">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                      </button>
                      <button onClick={() => resetTransform()} className="px-3 py-1 text-sm font-medium hover:bg-acid rounded-xl text-neutral-700 transition-colors" title="重置">
                        重置
                      </button>
                      <button onClick={() => zoomIn()} className="p-2 hover:bg-acid rounded-xl text-neutral-700 transition-colors" title="放大">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                      </button>
                    </div>
                    <TransformComponent wrapperClass="!w-full !h-full" contentClass="!w-full !h-full flex items-center justify-center">
                      <div className="w-full h-full flex items-center justify-center p-8">
                        <img src={resolvedSrc} alt={alt} className="max-w-full max-h-full object-contain" />
                      </div>
                    </TransformComponent>
                  </>
                )}
              </TransformWrapper>
            </div>
          </div>
        </div>
      ), document.body)}
    </>
  );
};

interface NotesTabProps {
  notes: Note[];
  targetId?: string;
  onReadingChange?: (reading: boolean) => void;
}

// Add Mermaid component
const MermaidRenderer = ({ chart }: { chart: string }) => {
  const [svg, setSvg] = useState<string>('');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const id = useMemo(() => `mermaid-${Math.random().toString(36).substr(2, 9)}`, []);
  const closeFullscreen = (event?: React.MouseEvent | React.PointerEvent) => {
    event?.preventDefault();
    event?.stopPropagation();
    setIsFullscreen(false);
  };

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
          className="flex justify-center overflow-x-auto bg-panel p-4 rounded-2xl border-2 border-neutral-900 shadow-[6px_6px_0_#17130f] transition-all duration-300 hover:border-accent cursor-zoom-in"
          onClick={() => setIsFullscreen(true)}
          dangerouslySetInnerHTML={{ __html: svg }} 
        />
        <button 
          onClick={() => setIsFullscreen(true)}
          className="absolute top-2 right-2 p-1.5 bg-acid backdrop-blur border border-neutral-900 rounded-xl text-neutral-900 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-accent hover:text-panel shadow-sm"
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

      {isFullscreen && createPortal((
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 backdrop-blur-sm p-4 sm:p-8 animate-in fade-in duration-200 cursor-zoom-out"
          onClick={() => setIsFullscreen(false)}
        >
          <div 
            className="relative w-[95vw] h-[95vh] bg-panel rounded-[2rem] shadow-2xl overflow-hidden cursor-default flex flex-col border-2 border-neutral-900"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="absolute top-4 right-4 z-[130] flex gap-2 pointer-events-auto">
              <button 
                type="button"
                onPointerDown={closeFullscreen}
                onMouseDown={(event) => event.stopPropagation()}
                onClick={closeFullscreen}
                className="p-2 bg-neutral-100 backdrop-blur text-neutral-400 hover:text-neutral-900 hover:bg-neutral-200 rounded-full transition-colors"
                title="关闭全屏"
                aria-label="关闭全屏"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>
            
            <div className="flex-1 w-full h-full bg-neutral-100">
              <TransformWrapper
                initialScale={1}
                minScale={0.1}
                maxScale={8}
                centerOnInit={true}
                wheel={{ step: 0.1 }}
              >
                {({ zoomIn, zoomOut, resetTransform }) => (
                  <>
                    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-panel/90 backdrop-blur-sm p-2 rounded-xl shadow-lg border border-neutral-200">
                      <button onClick={() => zoomOut()} className="p-2 hover:bg-neutral-100 rounded-lg text-neutral-700 transition-colors" title="缩小">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                      </button>
                      <button onClick={() => resetTransform()} className="px-3 py-1 text-sm font-medium hover:bg-neutral-100 rounded-lg text-neutral-700 transition-colors" title="重置">
                        重置
                      </button>
                      <button onClick={() => zoomIn()} className="p-2 hover:bg-neutral-100 rounded-lg text-neutral-700 transition-colors" title="放大">
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
      ), document.body)}
    </>
  );
};

const markdownComponents = {
  h1({children, ...props}: any) {
    return <h1 className="text-3xl font-extrabold mt-10 mb-4 text-neutral-900" {...props}>{children}</h1>
  },
  img({src, alt, ...props}: any) {
    return <ZoomableImage src={src} alt={alt} {...props} />;
  },
  pre({children}: any) {
    return <div className="not-prose">{children}</div>
  },
  blockquote({children, ...props}: any) {
    return (
      <blockquote className="note-callout" {...props}>
        <span className="note-callout-icon" aria-hidden="true">📌</span>
        <div className="note-callout-content">{children}</div>
      </blockquote>
    );
  },
  code({className, children, ...props}: any) {
    const match = /language-(\w+)/.exec(className || '')
    const isInline = !match && !String(children).includes('\n')
    if (isInline) {
      return (
        <code
          className="px-1.5 py-0.5 rounded-md text-[0.8em] font-mono break-words bg-[#dfead6] text-[#2f4a35] border border-[#b7c9ab]"
          {...props}
        >
          {children}
        </code>
      )
    }

    if (match && match[1] === 'mermaid') {
      return <MermaidRenderer chart={String(children).replace(/\n$/, '')} />;
    }

    const { ref, ...rest } = props;
    return (
      <div className="code-block rounded-2xl my-7 bg-[#0d0d0d] border-2 border-neutral-900 shadow-[8px_8px_0_#17130f] overflow-x-auto overflow-y-hidden">
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
            fontSize: '0.7em',
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
};

const MarkdownBlock = ({ content }: { content: string }) => (
  <ReactMarkdown
    remarkPlugins={[remarkGfm, remarkMath]}
    rehypePlugins={[rehypeRaw, rehypeRawMath, rehypeKatex, rehypeSlug, [rehypeAutolinkHeadings, { behavior: 'wrap' }]]}
    components={markdownComponents}
  >
    {content}
  </ReactMarkdown>
);

const MarkdownContent = ({ content }: { content: string }) => {
  const parts: React.ReactNode[] = [];
  const gridRe = /<grid>\s*([\s\S]*?)\s*<\/grid>/g;
  let lastIndex = 0;
  let gridMatch: RegExpExecArray | null;

  while ((gridMatch = gridRe.exec(content))) {
    const before = content.slice(lastIndex, gridMatch.index).trim();
    if (before) {
      parts.push(<MarkdownBlock key={`md-${parts.length}`} content={before} />);
    }

    const columns: Array<{ ratio?: string; content: string }> = [];
    const columnRe = /<column(?:\s+width-ratio="([^"]+)")?\s*>\s*([\s\S]*?)\s*<\/column>/g;
    let columnMatch: RegExpExecArray | null;
    while ((columnMatch = columnRe.exec(gridMatch[1]))) {
      columns.push({ ratio: columnMatch[1], content: columnMatch[2].trim() });
    }

    if (columns.length > 0) {
      parts.push(
        <div key={`grid-${parts.length}`} className="not-prose my-8 grid gap-6 lg:grid-flow-col lg:auto-cols-fr">
          {columns.map((column, index) => (
            <div
              key={index}
              className="min-w-0 prose prose-lg prose-site max-w-none prose-p:leading-8 prose-p:my-4 prose-li:leading-7 prose-li:my-1 prose-code:before:content-none prose-code:after:content-none prose-pre:p-0 prose-pre:m-0 prose-pre:bg-transparent prose-a:text-accent prose-a:no-underline hover:prose-a:underline"
              style={{ gridColumn: column.ratio ? `span ${Math.max(1, Math.round(Number(column.ratio) * 12))} / span ${Math.max(1, Math.round(Number(column.ratio) * 12))}` : undefined }}
            >
              <MarkdownBlock content={column.content} />
            </div>
          ))}
        </div>
      );
    }

    lastIndex = gridMatch.index + gridMatch[0].length;
  }

  const after = content.slice(lastIndex).trim();
  if (after) {
    parts.push(<MarkdownBlock key={`md-${parts.length}`} content={after} />);
  }

  return <>{parts}</>;
};

const NotesTab: React.FC<NotesTabProps> = ({ notes, targetId, onReadingChange }) => {
  const [active, setActive] = useState<Note | null>(() => (
    targetId ? notes.find(note => note.id === targetId) || null : null
  ));
  const [showToc, setShowToc] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const { notes: notesConfig } = siteConfig;

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
    setShowToc(false);
  }, [active]);

  // Report reading state so the app shell can collapse the top nav
  React.useEffect(() => {
    onReadingChange?.(!!active);
    return () => onReadingChange?.(false);
  }, [active, onReadingChange]);

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
      <div className="animate-in fade-in slide-in-from-right-4 duration-500 max-w-[90rem] mx-auto relative">
        <div className={`grid gap-6 isolate ${showToc ? 'lg:grid-cols-[18rem_minmax(0,1fr)]' : ''}`}>
          {showToc && (
            <aside className="relative z-30 self-start bg-panel rounded-[1.5rem] p-5 max-h-[calc(100vh-8rem)] overflow-hidden flex flex-col lg:sticky lg:top-28 shadow-[8px_8px_0_#17130f] animate-in fade-in slide-in-from-left-4 duration-200" style={{ border: '2px solid #17130f' }}>
              <div className="flex shrink-0 items-center justify-between mb-5">
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-accent">Article</p>
                  <h4 className="mt-1 text-lg font-display font-extrabold text-neutral-900 uppercase tracking-[-0.04em]">目录</h4>
                </div>
                <button
                  onClick={() => setShowToc(false)}
                  className="grid h-9 w-9 place-items-center rounded-full border border-neutral-900 bg-panel text-neutral-700 hover:bg-accent hover:text-panel transition-colors"
                  title="关闭目录"
                  aria-label="关闭目录"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                  </svg>
                </button>
              </div>
              <nav className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-2">
                {headings.length === 0 ? (
                  <p className="text-sm text-neutral-500 italic">暂无目录</p>
                ) : (
                  headings.map((heading) => (
                    <a
                      key={heading.id}
                      href={`#${heading.id}`}
                      className={`block rounded-xl px-3 py-2 text-sm transition-colors duration-200 hover:bg-acid/60 hover:text-neutral-900 ${
                        heading.level === 1 ? 'font-bold text-neutral-900' :
                        heading.level === 2 ? 'text-neutral-700' :
                        heading.level === 3 ? 'text-neutral-500' :
                        'text-neutral-400'
                      }`}
                      style={{
                        paddingLeft: heading.level > 1 ? `${(heading.level - 1) * 0.75 + 0.75}rem` : '0.75rem'
                      }}
                      onClick={(e) => {
                        e.preventDefault();
                        document.getElementById(heading.id)?.scrollIntoView({ behavior: 'smooth' });
                      }}
                    >
                      {heading.text}
                    </a>
                  ))
                )}
              </nav>
            </aside>
          )}

        <div className={`${showToc ? '' : 'max-w-5xl mx-auto'} relative z-10 studio-card rounded-[2rem] p-5 sm:p-8 lg:p-10 min-w-0 overflow-hidden`}>
          <div className="flex flex-col gap-5 pb-7 border-b-2 border-neutral-900 md:flex-row md:items-start md:justify-between">
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-3">
                <span className="rounded-full border border-neutral-900 bg-acid px-3 py-1 font-mono text-[10px] uppercase tracking-[0.22em] text-neutral-900">
                  {active.category}
                </span>
                <span className="text-xs text-neutral-500 font-mono">{active.createdAt}</span>
              </div>
              <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-[-0.06em] text-neutral-900">{active.title}</h1>
            </div>
            <div className="flex items-center gap-4">
              <button className="rounded-full border border-neutral-900 px-4 py-2 font-display text-xs uppercase tracking-[0.18em] text-neutral-700 hover:bg-accent hover:text-panel transition-colors" onClick={() => setActive(null)}>Back</button>
            </div>
          </div>
          <div className="prose prose-lg prose-site max-w-none pt-8 prose-p:leading-8 prose-p:my-4 prose-li:leading-7 prose-li:my-1 prose-code:before:content-none prose-code:after:content-none prose-pre:p-0 prose-pre:m-0 prose-pre:bg-transparent prose-a:text-accent prose-a:no-underline hover:prose-a:underline">
            <MarkdownContent content={active.content} />
        </div>
        </div>
        </div>

        <button
          onClick={() => setShowToc((visible) => !visible)}
          className={`fixed bottom-24 right-8 p-3 rounded-full shadow-[5px_5px_0_#17130f] border-2 border-neutral-900 hover:scale-110 transition-all duration-300 z-40 animate-in fade-in zoom-in duration-300 ${
            showToc ? 'bg-night text-panel' : 'bg-panel text-neutral-900 hover:bg-acid'
          }`}
          title={showToc ? '关闭目录' : '打开目录'}
          aria-label={showToc ? '关闭目录' : '打开目录'}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="4" y1="6" x2="20" y2="6"></line>
            <line x1="4" y1="12" x2="20" y2="12"></line>
            <line x1="4" y1="18" x2="14" y2="18"></line>
          </svg>
        </button>

        {/* Scroll to Top Button */}
        {showScrollTop && (
          <button
            onClick={scrollToTop}
            className="fixed bottom-8 right-8 p-3 bg-acid rounded-full shadow-[5px_5px_0_#17130f] border-2 border-neutral-900 text-neutral-900 hover:bg-accent hover:text-panel hover:scale-110 transition-all duration-300 z-40 animate-in fade-in zoom-in duration-300"
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
    <div className="space-y-10 animate-in fade-in slide-in-from-right-4 duration-500 max-w-7xl mx-auto">
      <div className="hard-card rounded-[2rem] p-7 sm:p-9 flex flex-col gap-6 md:flex-row md:justify-between md:items-end">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.26em] text-accent mb-4">{notesConfig.eyebrow}</p>
          <h1 className="hero-line text-6xl md:text-8xl font-extrabold uppercase text-neutral-900">{notesConfig.title}</h1>
          <p className="text-neutral-600 mt-5 max-w-xl leading-7">{notesConfig.description}</p>
        </div>
        <div className="rounded-full border-2 border-neutral-900 bg-acid px-5 py-3 font-mono text-sm uppercase tracking-[0.2em]">{cards.length} {notesConfig.countLabel}</div>
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
        {cards.map(({ note, cover, excerpt }, index) => (
          <button key={note.id} onClick={() => setActive(note)} className={`w-full min-h-[22rem] bg-panel rounded-[2rem] overflow-hidden border-2 border-neutral-900 group text-left card-hover flex flex-col ${index % 5 === 0 ? 'md:col-span-2' : ''}`}>
            {cover && (
              <div className="overflow-hidden border-b-2 border-neutral-900 max-h-72">
                <img src={resolveMarkdownAssetPath(cover)} alt={note.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
              </div>
            )}
            <div className="p-6 space-y-4 flex-1 flex flex-col">
              <div className="flex items-center gap-3">
                <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-neutral-900 bg-acid border border-neutral-900 px-3 py-1 rounded-full">{note.category}</span>
                <time className="text-xs text-neutral-500 font-mono">{note.createdAt}</time>
              </div>
              <h3 className="font-display text-3xl font-extrabold tracking-[-0.05em] text-neutral-900 group-hover:text-accent transition-colors">{note.title}</h3>
              <p className="text-neutral-600 text-sm leading-6 line-clamp-3">{excerpt}</p>
              <div className="flex gap-2 flex-wrap pt-2 mt-auto">
                {note.tags.map(tag => (
                  <span key={tag} className="text-[10px] font-mono uppercase tracking-wider text-neutral-500">#{tag}</span>
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

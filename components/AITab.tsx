
import React from 'react';

const links = [
  { name: 'ChatGPT', url: 'https://chat.openai.com/', desc: 'OpenAI 对话助手' },
  { name: 'Claude', url: 'https://claude.ai/', desc: 'Anthropic 智能写作助手' },
  { name: 'Gemini', url: 'https://gemini.google.com/', desc: 'Google 多模态助手' },
  { name: 'Perplexity', url: 'https://www.perplexity.ai/', desc: '检索增强问答' },
  { name: 'Midjourney', url: 'https://www.midjourney.com/', desc: 'AI 生成图片' },
  { name: 'Poe', url: 'https://poe.com/', desc: '多模型聚合平台' },
];

const AITab: React.FC = () => {
  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-top-4 duration-500">
      <div className="hard-card rounded-[2rem] p-7 sm:p-9 max-w-5xl">
        <p className="font-mono text-xs uppercase tracking-[0.26em] text-accent">Tools / quick launch</p>
        <h1 className="hero-line mt-4 text-6xl md:text-8xl font-extrabold uppercase text-neutral-900">
          AI 导航
        </h1>
        <p className="text-neutral-600 mt-5 max-w-xl leading-7">常用 AI 工具直达链接，点击在新标签页打开。这个页会更像一个 personal tool board。</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {links.map((link, index) => (
          <a
            key={link.name}
            href={link.url}
            target="_blank"
            rel="noopener noreferrer"
            className={`group min-h-[13rem] p-6 rounded-[2rem] border-2 border-neutral-900 bg-panel card-hover flex flex-col justify-between ${index === 0 ? 'lg:col-span-2 bg-acid' : ''}`}
          >
            <div className="flex items-center justify-between">
              <div className="w-12 h-12 rounded-full flex items-center justify-center text-panel bg-night group-hover:bg-accent group-hover:scale-110 transition-transform">
                <span className="text-xs font-display font-bold">AI</span>
              </div>
              <span className="text-xs font-mono text-neutral-500 group-hover:text-accent transition-colors">OPEN</span>
            </div>
            <div>
              <h3 className="font-display text-3xl font-extrabold tracking-[-0.05em] text-neutral-900 group-hover:text-accent transition-colors">{link.name}</h3>
              <p className="text-sm text-neutral-600 mt-2 leading-6">{link.desc}</p>
            </div>
          </a>
        ))}
      </div>
    </div>
  );
};

export default AITab;


import React from 'react';

const links = [
  { name: 'ChatGPT', url: 'https://chat.openai.com/', color: 'indigo', desc: 'OpenAI 对话助手' },
  { name: 'Claude', url: 'https://claude.ai/', color: 'amber', desc: 'Anthropic 智能写作助手' },
  { name: 'Gemini', url: 'https://gemini.google.com/', color: 'emerald', desc: 'Google 多模态助手' },
  { name: 'Perplexity', url: 'https://www.perplexity.ai/', color: 'cyan', desc: '检索增强问答' },
  { name: 'Midjourney', url: 'https://www.midjourney.com/', color: 'rose', desc: 'AI 生成图片' },
  { name: 'Poe', url: 'https://poe.com/', color: 'violet', desc: '多模型聚合平台' }
];

const AITab: React.FC = () => {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-top-4 duration-500">
      <div className="max-w-3xl">
        <h1 className="text-3xl font-bold text-slate-900">AI 工具导航</h1>
        <p className="text-slate-500">常用 AI 工具直达链接，点击在新标签页打开。</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {links.map(link => (
          <a
            key={link.name}
            href={link.url}
            target="_blank"
            rel="noopener noreferrer"
            className="group p-6 bg-white rounded-2xl border border-slate-100 hover:shadow-xl transition-all flex items-start gap-4"
          >
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-white bg-${link.color}-500 group-hover:scale-110 transition-transform`}>
              <span className="text-sm font-black">AI</span>
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-slate-900">{link.name}</h3>
                <span className="text-xs text-slate-400 group-hover:text-slate-600">↗</span>
              </div>
              <p className="text-sm text-slate-500 mt-1">{link.desc}</p>
            </div>
          </a>
        ))}
      </div>
    </div>
  );
};

export default AITab;

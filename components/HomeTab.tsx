
import React from 'react';
import { AppTab } from '../types';

interface HomeTabProps {
  onNavigate: (tab: AppTab) => void;
  notesCount: number;
  momentsCount: number;
}

const HomeTab: React.FC<HomeTabProps> = ({ onNavigate, notesCount, momentsCount }) => {
  return (
    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header className="space-y-4">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight">
          你好，<span className="text-indigo-600">探险家</span>
        </h1>
        <p className="text-xl text-slate-500 max-w-2xl leading-relaxed">
          这里是你的数字绿洲。在这个喧嚣的互联网时代，有一个只属于自己的空间去沉淀知识和记录生活是弥足珍贵的。
        </p>
      </header>

      {/* 核心看板 */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div 
          onClick={() => onNavigate(AppTab.NOTES)}
          className="group p-8 bg-white rounded-3xl border border-slate-100 shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all cursor-pointer"
        >
          <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-blue-600 group-hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold mb-2">知识库</h3>
          <p className="text-slate-500 mb-4">已有 {notesCount} 篇沉淀。知识只有被记录和整理，才能真正成为你的力量。</p>
          <div className="text-blue-600 font-semibold flex items-center gap-2">
            进入笔记库 <span className="text-lg">→</span>
          </div>
        </div>

        <div 
          onClick={() => onNavigate(AppTab.LIFE)}
          className="group p-8 bg-white rounded-3xl border border-slate-100 shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all cursor-pointer"
        >
          <div className="w-12 h-12 bg-rose-50 text-rose-600 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-rose-600 group-hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold mb-2">生活志</h3>
          <p className="text-slate-500 mb-4">记录了 {momentsCount} 个瞬间。生活不是你活过的日子，而是你记住的日子。</p>
          <div className="text-rose-600 font-semibold flex items-center gap-2">
            浏览瞬间 <span className="text-lg">→</span>
          </div>
        </div>

        <div 
          onClick={() => onNavigate(AppTab.AI)}
          className="group p-8 bg-indigo-600 rounded-3xl shadow-lg shadow-indigo-200 hover:shadow-2xl hover:-translate-y-1 transition-all cursor-pointer text-white"
        >
          <div className="w-12 h-12 bg-white/20 rounded-2xl flex items-center justify-center mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold mb-2">AI 创意助手</h3>
          <p className="text-indigo-100 mb-4">没灵感了？让 Gemini 帮你总结笔记，或者为你推荐下个写作题目。</p>
          <div className="text-white font-semibold flex items-center gap-2">
            开启灵感 <span className="text-lg">→</span>
          </div>
        </div>
      </section>

      {/* 专业的双分支管理方案 */}
      <section className="bg-slate-900 rounded-[2.5rem] p-8 md:p-12 text-white relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 blur-[100px] -mr-48 -mt-48"></div>
        
        <div className="relative z-10 space-y-10">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h2 className="text-3xl font-bold">双分支协作管理方案</h2>
              <p className="text-slate-400 mt-2">将源代码与展示页面分离，像专家一样管理你的主页。</p>
            </div>
            <div className="flex gap-3">
              <div className="px-4 py-1.5 bg-indigo-500/20 text-indigo-400 rounded-lg border border-indigo-500/30 text-xs font-mono">
                Source: source branch
              </div>
              <div className="px-4 py-1.5 bg-emerald-500/20 text-emerald-400 rounded-lg border border-emerald-500/30 text-xs font-mono">
                Deploy: master branch
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* 左侧：日常开发流程 */}
            <div className="space-y-6">
              <div className="flex items-center gap-3 text-indigo-400">
                <div className="w-8 h-8 rounded-full bg-indigo-500/20 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold">1. 日常开发 (Source 分支)</h3>
              </div>
              <p className="text-slate-400 text-sm">
                你的 React 源代码、组件、所有的“原材料”都应该保存在这里。这是你的备份中心。
              </p>
              <div className="bg-black/40 rounded-2xl p-5 font-mono text-xs text-indigo-300 border border-white/5 space-y-2">
                <p># 确保你在 source 分支</p>
                <p>git checkout -b source</p>
                <p># 修改代码后备份</p>
                <p>git add .</p>
                <p>git commit -m "Update oasis"</p>
                <p>git push origin source</p>
              </div>
            </div>

            {/* 右侧：发布流程 */}
            <div className="space-y-6">
              <div className="flex items-center gap-3 text-emerald-400">
                <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold">2. 发布上线 (Master 分支)</h3>
              </div>
              <p className="text-slate-400 text-sm">
                通过脚本将编译后的“成品”发布到 master。它不会影响你的 source 分支内容。
              </p>
              <div className="bg-black/40 rounded-2xl p-5 font-mono text-xs text-emerald-300 border border-emerald-500/20 space-y-4">
                <div>
                  <p># 一键编译并发布到 master</p>
                  <p className="text-white mt-1">npm run deploy</p>
                </div>
                <div className="pt-2 border-t border-white/5">
                  <p className="text-slate-500 italic"># 发布后，GitHub 会在 30 秒内自动更新：</p>
                  <p className="text-emerald-400">https://lucienne999.github.io/</p>
                </div>
              </div>
            </div>
          </div>

          {/* 底部提示 */}
          <div className="bg-white/5 rounded-3xl p-6 border border-white/10 flex items-center gap-4">
            <div className="flex-shrink-0 w-12 h-12 rounded-2xl bg-amber-500/20 flex items-center justify-center text-amber-500">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-bold text-slate-200">无需手动切换分支发布</p>
              <p className="text-xs text-slate-400 mt-1">
                已经在 <code className="text-indigo-300">package.json</code> 中配置好了：当你运行 <code className="text-indigo-300">npm run deploy</code> 时，系统会自动把结果送往 <code className="text-indigo-300">master</code>，你只需要安心在 <code className="text-indigo-300">source</code> 分支写代码即可。
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomeTab;

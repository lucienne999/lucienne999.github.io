
import React, { useState } from 'react';
import { Note, Suggestion } from '../types';
import { summarizeNotes, generateWritingPrompts } from '../services/geminiService';

interface AITabProps {
  notes: Note[];
}

const AITab: React.FC<AITabProps> = ({ notes }) => {
  const [summary, setSummary] = useState<string | null>(null);
  const [prompts, setPrompts] = useState<Suggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [context, setContext] = useState('');

  const handleSummarize = async () => {
    setLoading(true);
    try {
      const res = await summarizeNotes(notes);
      setSummary(res || "总结失败，请重试。");
    } catch (e) {
      console.error(e);
      setSummary("连接 AI 失败。");
    } finally {
      setLoading(false);
    }
  };

  const handleGetPrompts = async () => {
    if (!context) return;
    setLoading(true);
    try {
      const res = await generateWritingPrompts(context);
      setPrompts(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-top-4 duration-500">
      <div className="max-w-3xl">
        <h1 className="text-3xl font-bold text-slate-900">AI 创意助手</h1>
        <p className="text-slate-500">让 Gemini 助你一臂之力，挖掘笔记中的深度联系，寻找下个灵感。</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <section className="bg-white p-8 rounded-[2rem] border border-slate-100 space-y-6 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-100 text-indigo-600 rounded-xl flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
            <h2 className="text-xl font-bold">笔记全站总结</h2>
          </div>
          <p className="text-sm text-slate-500">
            分析你现有的 {notes.length} 篇笔记，找出隐藏的主题和规律。
          </p>
          <button 
            onClick={handleSummarize}
            disabled={loading}
            className="w-full bg-slate-900 text-white py-3 rounded-xl font-bold hover:bg-slate-800 disabled:opacity-50"
          >
            {loading ? '正在思考...' : '开始分析'}
          </button>
          
          {summary && (
            <div className="p-5 bg-slate-50 rounded-2xl border border-slate-100 prose prose-slate text-sm">
              <p className="whitespace-pre-wrap">{summary}</p>
            </div>
          )}
        </section>

        <section className="bg-white p-8 rounded-[2rem] border border-slate-100 space-y-6 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-amber-100 text-amber-600 rounded-xl flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <h2 className="text-xl font-bold">写作启发器</h2>
          </div>
          <p className="text-sm text-slate-500">
            告诉我你目前感兴趣的话题或情绪，AI 将为你推荐写作题目。
          </p>
          <div className="flex gap-2">
            <input 
              className="flex-1 bg-slate-50 border border-slate-100 px-4 py-2 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none"
              placeholder="例如：对咖啡文化的热爱，或者前端性能优化..."
              value={context}
              onChange={(e) => setContext(e.target.value)}
            />
            <button 
              onClick={handleGetPrompts}
              disabled={loading || !context}
              className="bg-amber-500 text-white px-4 py-2 rounded-xl font-bold hover:bg-amber-600 disabled:opacity-50"
            >
              获取
            </button>
          </div>

          <div className="space-y-4">
            {prompts.map((p, idx) => (
              <div key={idx} className="p-4 bg-amber-50 rounded-2xl border border-amber-100 hover:border-amber-200 transition-colors">
                <h4 className="font-bold text-amber-900 mb-1">主题：{p.topic}</h4>
                <p className="text-xs text-amber-700">推荐理由：{p.reason}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};

export default AITab;

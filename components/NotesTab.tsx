
import React, { useState } from 'react';
import { Note } from '../types';
import { ICONS } from '../constants';

interface NotesTabProps {
  notes: Note[];
  onAddNote: (note: Omit<Note, 'id' | 'createdAt'>) => void;
}

const NotesTab: React.FC<NotesTabProps> = ({ notes, onAddNote }) => {
  const [isAdding, setIsAdding] = useState(false);
  const [newNote, setNewNote] = useState({ title: '', content: '', category: '随笔', tags: [] as string[] });
  const [tagInput, setTagInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newNote.title || !newNote.content) return;
    onAddNote(newNote);
    setNewNote({ title: '', content: '', category: '随笔', tags: [] });
    setIsAdding(false);
  };

  const addTag = () => {
    if (tagInput.trim() && !newNote.tags.includes(tagInput.trim())) {
      setNewNote({ ...newNote, tags: [...newNote.tags, tagInput.trim()] });
      setTagInput('');
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-right-4 duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">知识仓库</h1>
          <p className="text-slate-500">捕捉灵感，归档见解。</p>
        </div>
        <button 
          onClick={() => setIsAdding(true)}
          className="bg-indigo-600 text-white px-5 py-2.5 rounded-xl font-medium flex items-center gap-2 hover:bg-indigo-700 transition-colors shadow-lg shadow-indigo-100"
        >
          <ICONS.Add /> 新建笔记
        </button>
      </div>

      {isAdding && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-[60] flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl w-full max-w-2xl overflow-hidden shadow-2xl animate-in zoom-in-95 duration-200">
            <form onSubmit={handleSubmit} className="p-8 space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold">写点什么...</h2>
                <button type="button" onClick={() => setIsAdding(false)} className="text-slate-400 hover:text-slate-600">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="space-y-4">
                <input
                  autoFocus
                  className="w-full text-2xl font-bold focus:outline-none placeholder:text-slate-300"
                  placeholder="标题"
                  value={newNote.title}
                  onChange={(e) => setNewNote({ ...newNote, title: e.target.value })}
                />
                <select 
                  className="bg-slate-100 px-3 py-1.5 rounded-lg text-sm font-medium focus:ring-2 focus:ring-indigo-500 outline-none"
                  value={newNote.category}
                  onChange={(e) => setNewNote({ ...newNote, category: e.target.value })}
                >
                  <option>随笔</option>
                  <option>技术</option>
                  <option>读书</option>
                  <option>项目</option>
                </select>
                <textarea
                  className="w-full h-48 focus:outline-none resize-none placeholder:text-slate-300 text-slate-700"
                  placeholder="在此处开始你的思考..."
                  value={newNote.content}
                  onChange={(e) => setNewNote({ ...newNote, content: e.target.value })}
                />
              </div>

              <div className="space-y-3">
                <div className="flex flex-wrap gap-2">
                  {newNote.tags.map(tag => (
                    <span key={tag} className="px-3 py-1 bg-indigo-50 text-indigo-600 rounded-full text-xs font-semibold">
                      #{tag}
                    </span>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input
                    className="flex-1 border border-slate-200 rounded-lg px-4 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                    placeholder="添加标签..."
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                  />
                  <button type="button" onClick={addTag} className="px-4 py-2 text-indigo-600 font-medium hover:bg-indigo-50 rounded-lg">添加</button>
                </div>
              </div>

              <button type="submit" className="w-full bg-indigo-600 text-white py-4 rounded-2xl font-bold hover:bg-indigo-700 transition-colors shadow-lg shadow-indigo-200">
                发布笔记
              </button>
            </form>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {notes.map(note => (
          <article key={note.id} className="bg-white p-8 rounded-3xl border border-slate-100 hover:border-indigo-100 transition-all group">
            <div className="flex justify-between items-start mb-4">
              <span className="text-xs font-bold uppercase tracking-widest text-indigo-600 bg-indigo-50 px-3 py-1 rounded-full">
                {note.category}
              </span>
              <time className="text-xs text-slate-400 font-medium">{note.createdAt}</time>
            </div>
            <h3 className="text-xl font-bold mb-3 group-hover:text-indigo-600 transition-colors">{note.title}</h3>
            <p className="text-slate-600 line-clamp-3 mb-6 leading-relaxed">{note.content}</p>
            <div className="flex flex-wrap gap-2">
              {note.tags.map(tag => (
                <span key={tag} className="text-xs text-slate-400">#{tag}</span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
};

export default NotesTab;

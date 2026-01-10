
import React, { useState } from 'react';
import { LifeMoment } from '../types';
import { ICONS } from '../constants';

interface LifeTabProps {
  moments: LifeMoment[];
  onAddMoment: (moment: Omit<LifeMoment, 'id' | 'date'>) => void;
}

const LifeTab: React.FC<LifeTabProps> = ({ moments, onAddMoment }) => {
  const [isAdding, setIsAdding] = useState(false);
  const [newMoment, setNewMoment] = useState({ description: '', location: '', imageUrl: '' });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMoment.description) return;
    
    // Default image if none provided
    const imageUrl = newMoment.imageUrl || `https://picsum.photos/seed/${Date.now()}/800/600`;
    onAddMoment({ ...newMoment, imageUrl });
    setNewMoment({ description: '', location: '', imageUrl: '' });
    setIsAdding(false);
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-left-4 duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">生活志</h1>
          <p className="text-slate-500">记录那些闪闪发光的瞬间。</p>
        </div>
        <button 
          onClick={() => setIsAdding(true)}
          className="bg-rose-500 text-white px-5 py-2.5 rounded-xl font-medium flex items-center gap-2 hover:bg-rose-600 transition-colors shadow-lg shadow-rose-100"
        >
          <ICONS.Add /> 记录瞬间
        </button>
      </div>

      {isAdding && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-[60] flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl w-full max-w-lg overflow-hidden shadow-2xl animate-in zoom-in-95 duration-200">
            <form onSubmit={handleSubmit} className="p-8 space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-rose-600">捕捉瞬间</h2>
                <button type="button" onClick={() => setIsAdding(false)} className="text-slate-400 hover:text-slate-600">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="space-y-4">
                <textarea
                  autoFocus
                  className="w-full h-32 focus:outline-none resize-none placeholder:text-slate-300 text-lg"
                  placeholder="这一刻发生了什么..."
                  value={newMoment.description}
                  onChange={(e) => setNewMoment({ ...newMoment, description: e.target.value })}
                />
                <input
                  className="w-full border-b border-slate-100 py-2 focus:outline-none focus:border-rose-300"
                  placeholder="你在哪里？"
                  value={newMoment.location}
                  onChange={(e) => setNewMoment({ ...newMoment, location: e.target.value })}
                />
                <input
                  className="w-full border-b border-slate-100 py-2 focus:outline-none focus:border-rose-300"
                  placeholder="图片 URL (可选)"
                  value={newMoment.imageUrl}
                  onChange={(e) => setNewMoment({ ...newMoment, imageUrl: e.target.value })}
                />
              </div>

              <button type="submit" className="w-full bg-rose-500 text-white py-4 rounded-2xl font-bold hover:bg-rose-600 transition-colors">
                保存瞬间
              </button>
            </form>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {moments.map(moment => (
          <div key={moment.id} className="bg-white rounded-[2rem] overflow-hidden border border-slate-100 group">
            <div className="aspect-[4/3] overflow-hidden relative">
              <img 
                src={moment.imageUrl} 
                alt={moment.description} 
                className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
              />
              <div className="absolute top-4 right-4 bg-white/90 backdrop-blur px-3 py-1 rounded-full text-[10px] font-bold text-slate-900 uppercase">
                {moment.date}
              </div>
            </div>
            <div className="p-6 space-y-3">
              <p className="text-slate-700 font-medium leading-relaxed">{moment.description}</p>
              {moment.location && (
                <div className="flex items-center gap-1.5 text-slate-400 text-xs">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                  </svg>
                  {moment.location}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LifeTab;

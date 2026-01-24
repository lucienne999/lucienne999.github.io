
import React, { useState, useEffect } from 'react';
import { AppTab, Note, LifeMoment } from './types';
import { ICONS, INITIAL_NOTES, INITIAL_MOMENTS } from './constants';
import { loadNotes, loadMoments } from './services/contentLoader';
import HomeTab from './components/HomeTab';
import NotesTab from './components/NotesTab';
import LifeTab from './components/LifeTab';
import AITab from './components/AITab';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AppTab>(AppTab.HOME);
  const [notes, setNotes] = useState<Note[]>(INITIAL_NOTES);
  const [moments, setMoments] = useState<LifeMoment[]>(INITIAL_MOMENTS);

  useEffect(() => {
    try {
      const n = loadNotes();
      const m = loadMoments();
      console.log('Loaded notes:', n);
      console.log('Loaded moments:', m);
      setNotes(n);
      setMoments(m);
    } catch (e) {
      console.error('Failed to load content:', e);
    }
  }, []);


  const renderContent = () => {
    switch (activeTab) {
      case AppTab.HOME:
        return <HomeTab onNavigate={setActiveTab} notes={notes} moments={moments} />;
      case AppTab.NOTES:
        return <NotesTab notes={notes} />;
      case AppTab.LIFE:
        return <LifeTab moments={moments} />;
      case AppTab.AI:
        return <AITab />;
      default:
        return <HomeTab onNavigate={setActiveTab} notes={notes} moments={moments} />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-24 md:pb-0 md:pl-20">
      {/* Sidebar Navigation (Desktop) */}
      <nav className="hidden md:flex fixed left-0 top-0 h-full w-20 bg-white border-r border-slate-200 flex-col items-center py-8 gap-8 z-50">
        <NavButton active={activeTab === AppTab.HOME} onClick={() => setActiveTab(AppTab.HOME)} icon={<ICONS.Home />} label="首页" />
        <NavButton active={activeTab === AppTab.NOTES} onClick={() => setActiveTab(AppTab.NOTES)} icon={<ICONS.Notes />} label="笔记" />
        <NavButton active={activeTab === AppTab.LIFE} onClick={() => setActiveTab(AppTab.LIFE)} icon={<ICONS.Life />} label="生活" />
        <NavButton active={activeTab === AppTab.AI} onClick={() => setActiveTab(AppTab.AI)} icon={<ICONS.AI />} label="智囊" />
      </nav>

      {/* Bottom Navigation (Mobile) */}
      <nav className="md:hidden fixed bottom-0 left-0 w-full bg-white border-t border-slate-200 flex justify-around items-center py-4 z-50 glass-panel">
        <NavButton active={activeTab === AppTab.HOME} onClick={() => setActiveTab(AppTab.HOME)} icon={<ICONS.Home />} label="首页" isMobile />
        <NavButton active={activeTab === AppTab.NOTES} onClick={() => setActiveTab(AppTab.NOTES)} icon={<ICONS.Notes />} label="笔记" isMobile />
        <NavButton active={activeTab === AppTab.LIFE} onClick={() => setActiveTab(AppTab.LIFE)} icon={<ICONS.Life />} label="生活" isMobile />
        <NavButton active={activeTab === AppTab.AI} onClick={() => setActiveTab(AppTab.AI)} icon={<ICONS.AI />} label="智囊" isMobile />
      </nav>

      {/* Main Content Area */}
      <main className="max-w-6xl mx-auto px-6 py-8 md:py-12">
        {renderContent()}
      </main>
    </div>
  );
};

interface NavButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  isMobile?: boolean;
}

const NavButton: React.FC<NavButtonProps> = ({ active, onClick, icon, label, isMobile }) => {
  return (
    <button
      onClick={onClick}
      className={`flex flex-col items-center gap-1 transition-all duration-300 ${
        active ? 'text-indigo-600 scale-110' : 'text-slate-400 hover:text-slate-600'
      }`}
    >
      <div className={`p-2 rounded-lg ${active && !isMobile ? 'bg-indigo-50' : ''}`}>
        {icon}
      </div>
      <span className="text-[10px] font-medium uppercase tracking-wider">{label}</span>
    </button>
  );
};

export default App;

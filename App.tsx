
import React, { useState, useEffect, useCallback } from 'react';
import { AppTab, Note, LifeMoment } from './types';
import { ICONS, INITIAL_NOTES, INITIAL_MOMENTS } from './constants';
import { loadNotes, loadMoments } from './services/contentLoader';
import HomeTab from './components/HomeTab';
import NotesTab from './components/NotesTab';
import LifeTab from './components/LifeTab';
import { siteConfig } from './config/siteConfig';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AppTab>(AppTab.HOME);
  const [targetId, setTargetId] = useState<string | undefined>(undefined);
  const [notes, setNotes] = useState<Note[]>(INITIAL_NOTES);
  const [moments, setMoments] = useState<LifeMoment[]>(INITIAL_MOMENTS);
  const [isReadingNote, setIsReadingNote] = useState(false);

  const handleReadingChange = useCallback((reading: boolean) => {
    setIsReadingNote(reading);
  }, []);

  useEffect(() => {
    try {
      const n = loadNotes();
      const m = loadMoments();
      setNotes(n);
      setMoments(m);
    } catch (e) {
      console.error('Failed to load content:', e);
    }
  }, []);

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1);
      if (!hash) {
        setActiveTab(AppTab.HOME);
        setTargetId(undefined);
        return;
      }

      const [tab, id] = hash.split('/');
      const tabMap: Record<string, AppTab> = {
        home: AppTab.HOME,
        notes: AppTab.NOTES,
        life: AppTab.LIFE,
      };

      if (tabMap[tab]) {
        setActiveTab(tabMap[tab]);
        setTargetId(id ? decodeURIComponent(id) : undefined);
      }
    };

    handleHashChange();
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const handleNavigate = (tab: AppTab, id?: string) => {
    setActiveTab(tab);
    setTargetId(id);

    let hash = '';
    switch (tab) {
      case AppTab.HOME: hash = 'home'; break;
      case AppTab.NOTES: hash = 'notes'; break;
      case AppTab.LIFE: hash = 'life'; break;
    }

    if (id) {
      hash += `/${encodeURIComponent(id)}`;
    }

    window.location.hash = hash;
  };

  const renderContent = () => {
    switch (activeTab) {
      case AppTab.HOME:
        return <HomeTab onNavigate={handleNavigate} notes={notes} moments={moments} />;
      case AppTab.NOTES:
        return <NotesTab notes={notes} targetId={targetId} onReadingChange={handleReadingChange} />;
      case AppTab.LIFE:
        return <LifeTab moments={moments} targetId={targetId} />;
      default:
        return <HomeTab onNavigate={handleNavigate} notes={notes} moments={moments} />;
    }
  };

  const navItems = [
    { tab: AppTab.HOME, label: 'Home', icon: <ICONS.Home /> },
    { tab: AppTab.NOTES, label: 'Notes', icon: <ICONS.Notes /> },
    { tab: AppTab.LIFE, label: 'Life', icon: <ICONS.Life /> },
  ];

  const { brand } = siteConfig;

  return (
    <div className="min-h-screen site-shell text-neutral-900 pb-24 md:pb-0">
      <header className={`fixed z-50 transition-all duration-500 ${isReadingNote ? 'left-3 top-6 px-0' : 'left-0 right-0 top-4 px-4'}`}>
        <div className={`glass-panel flex transition-all duration-500 ${
          isReadingNote
            ? 'w-14 flex-col items-center gap-2 rounded-[2rem] px-2 py-3'
            : 'max-w-7xl mx-auto items-center justify-between rounded-full px-4 md:px-5 py-3'
        }`}>
          <button
            onClick={() => handleNavigate(AppTab.HOME)}
            className={`group flex items-center ${isReadingNote ? 'flex-col gap-1' : 'gap-3'}`}
            title="回到首页"
            aria-label="回到首页"
          >
            <span className={`grid place-items-center rounded-full bg-night text-panel font-display font-bold group-hover:bg-accent transition-colors ${
              isReadingNote ? 'h-9 w-9 text-xs' : 'h-10 w-10 text-sm'
            }`}>
              {brand.initials}
            </span>
            <span className={`text-left transition-all duration-300 ${isReadingNote ? 'hidden' : 'hidden sm:block'}`}>
              <span className="block font-display text-sm uppercase tracking-[0.22em] leading-none">{brand.name}</span>
              <span className="block text-[10px] uppercase tracking-[0.28em] text-neutral-500 mt-1">{brand.tagline}</span>
            </span>
          </button>

          {isReadingNote ? (
            <nav className="flex flex-col items-center gap-1 border-t border-neutral-900/10 pt-2">
              {navItems.map(({ tab, label, icon }) => (
                <SideNavButton
                  key={label}
                  active={activeTab === tab}
                  label={label}
                  icon={icon}
                  onClick={() => handleNavigate(tab)}
                />
              ))}
            </nav>
          ) : (
            <nav className="hidden md:flex items-center gap-2 rounded-full bg-white/45 p-1 border border-neutral-900/10">
              {navItems.map(({ tab, label }) => (
                <NavLink
                  key={label}
                  active={activeTab === tab}
                  label={label}
                  onClick={() => handleNavigate(tab)}
                />
              ))}
            </nav>
          )}
        </div>
      </header>

      <nav className="md:hidden fixed bottom-4 left-4 right-4 glass-panel rounded-[1.75rem] flex justify-around items-center py-3 z-50">
        {navItems.map(({ tab, label, icon }) => (
          <NavButton
            key={label}
            active={activeTab === tab}
            onClick={() => handleNavigate(tab)}
            icon={icon}
            label={label}
            isMobile
          />
        ))}
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 pt-28 pb-8 md:pt-32 md:pb-20">
        {renderContent()}
      </main>
    </div>
  );
};

interface NavLinkProps {
  active: boolean;
  label: string;
  onClick: () => void;
}

const NavLink: React.FC<NavLinkProps> = ({ active, label, onClick }) => (
  <button
    onClick={onClick}
    className={`font-display text-xs uppercase tracking-[0.18em] transition-all flex items-center gap-2 rounded-full px-4 py-2 ${
      active ? 'bg-night text-panel shadow-sm' : 'text-neutral-600 hover:text-neutral-900 hover:bg-white/70'
    }`}
  >
    {active && <span className="inline-block w-1.5 h-1.5 rounded-full bg-acid nav-dot" />}
    {label}
  </button>
);

interface NavButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  isMobile?: boolean;
}

const NavButton: React.FC<NavButtonProps> = ({ active, onClick, icon, label }) => (
  <button
    onClick={onClick}
    className={`flex flex-col items-center gap-1 transition-all duration-300 ${
      active ? 'text-neutral-900 scale-105' : 'text-neutral-500'
    }`}
  >
    <div className={`p-2 rounded-2xl ${active ? 'bg-acid border border-neutral-900' : ''}`}>{icon}</div>
    <span className="font-display text-[10px] font-medium uppercase tracking-wider">{label}</span>
  </button>
);

const SideNavButton: React.FC<NavButtonProps> = ({ active, onClick, icon, label }) => (
  <button
    onClick={onClick}
    title={label}
    aria-label={label}
    className={`grid h-9 w-9 place-items-center rounded-full transition-all duration-300 ${
      active
        ? 'bg-acid text-neutral-900 border border-neutral-900 shadow-sm'
        : 'text-neutral-500 hover:text-neutral-900 hover:bg-white/70'
    }`}
  >
    {icon}
  </button>
);

export default App;

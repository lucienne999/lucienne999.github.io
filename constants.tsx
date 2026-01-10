
import React from 'react';
import { Note, LifeMoment } from './types';

export const INITIAL_NOTES: Note[] = [
  {
    id: '1',
    title: '关于搭建个人网站的思考',
    content: '个人网站不仅仅是展示，更是思维的归档。第一步应该是确定核心价值：是作为作品集，还是作为长期的知识库？',
    tags: ['建站', '思考'],
    createdAt: '2024-05-20',
    category: '随笔'
  },
  {
    id: '2',
    title: 'React 18 并发渲染',
    content: '深入理解 StartTransition 和 UseDeferredValue 的应用场景...',
    tags: ['React', '前端'],
    createdAt: '2024-05-18',
    category: '技术'
  }
];

export const INITIAL_MOMENTS: LifeMoment[] = [
  {
    id: 'm1',
    imageUrl: 'https://picsum.photos/seed/nature/800/600',
    description: '周末在郊外徒步，空气很好。',
    date: '2024-05-19',
    location: '奥森公园'
  },
  {
    id: 'm2',
    imageUrl: 'https://picsum.photos/seed/coffee/800/600',
    description: '尝试了一家新开的独立咖啡店，手冲味道很惊艳。',
    date: '2024-05-21',
    location: '静安区'
  }
];

export const ICONS = {
  Home: () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
    </svg>
  ),
  Notes: () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
    </svg>
  ),
  Life: () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  ),
  AI: () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  Add: () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
  )
};

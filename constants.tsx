
import React from 'react';
import { Note, LifeMoment } from './types';

// 自动读取 posts 目录下的所有 .md 文件
// 注意：这需要 Vite 环境支持。在本地开发时，请确保创建了 posts 文件夹
const markdownFiles = import.meta.glob('./posts/*.md', { query: '?raw', eager: true });

const autoLoadedNotes: Note[] = Object.entries(markdownFiles).map(([path, content]) => {
  // 从路径中提取文件名作为标题，例如 "./posts/我的笔记.md" -> "我的笔记"
  const fileName = path.split('/').pop()?.replace('.md', '') || '未命名文章';
  
  return {
    id: path,
    title: fileName,
    category: '外部源文件',
    createdAt: new Date().toISOString().split('T')[0], // 实际生产中建议在 Markdown Frontmatter 中定义日期
    tags: ['自动同步'],
    content: (content as any).default || content as string
  };
});

export const INITIAL_NOTES: Note[] = [
  ...autoLoadedNotes,
  {
    id: 'welcome-oasis',
    title: '欢迎来到你的数字绿洲',
    category: '随笔',
    createdAt: '2024-05-23',
    tags: ['指南'],
    content: `这是你的第一篇手动笔记。

你现在可以通过直接在项目根目录下的 \`posts\` 文件夹中放置 \`.md\` 文件来增加文章。
系统会自动识别它们并显示在“知识仓库”中。`
  }
];

export const INITIAL_MOMENTS: LifeMoment[] = [
  {
    id: 'm1',
    imageUrl: 'https://images.unsplash.com/photo-1501785888041-af3ef285b470?q=80&w=2070&auto=format&fit=crop',
    description: '周末在郊外徒步，空气很好。',
    content: '周末在郊外徒步，空气很好。这是初始数据。',
    date: '2024-05-19',
    location: '奥森公园'
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
  Guestbook: () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
    </svg>
  ),
  Add: () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
  )
};

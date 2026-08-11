export const siteConfig = {
  brand: {
    initials: 'Hi',
    name: 'Lucienne',
    tagline: 'notes / life ',
  },

  home: {
    badge: 'Personal Blog',
    heroTitle: ['Being', 'toward', 'Death'],
    heroDescription: '这个人很懒，也想写点什么。',
    tickerText: '',
    notesCard: {
      eyebrow: 'Archive',
      title: 'Notes',
      description: '技术、系统、模型和工程笔记。',
    },
    lifeCard: {
      eyebrow: 'Moments',
      title: 'Life',
      description: '照片、地点和一些偶然记录。',
    },
  },

  notes: {
    eyebrow: 'Archive / writing system',
    title: 'Notes',
    description: '日拱一卒。',
    countLabel: 'posts',
  },

  life: {
    eyebrow: 'Moments / field notes',
    title: 'Life',
    description: '记录那些闪闪发光的瞬间。',
    countLabel: 'moments',
  },
} as const;


export enum AppTab {
  HOME = 'home',
  NOTES = 'notes',
  LIFE = 'life',
  AI = 'ai',
  GUESTBOOK = 'guestbook'
}

export interface Note {
  id: string;
  title: string;
  content: string;
  tags: string[];
  createdAt: string;
  category: string;
}

export interface LifeMoment {
  id: string;
  title?: string;
  imageUrl?: string;
  description: string;
  content: string;
  date: string;
  location?: string;
  tags?: string[];
}

export interface Suggestion {
  topic: string;
  reason: string;
}

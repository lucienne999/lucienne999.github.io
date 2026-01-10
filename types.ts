
export enum AppTab {
  HOME = 'home',
  NOTES = 'notes',
  LIFE = 'life',
  AI = 'ai'
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
  imageUrl: string;
  description: string;
  date: string;
  location?: string;
}

export interface Suggestion {
  topic: string;
  reason: string;
}

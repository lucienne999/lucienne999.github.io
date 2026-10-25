import { Note, LifeMoment } from '../types';

type Frontmatter = Record<string, string | string> & {
  [key: string]: any;
};

function parseFrontmatter(md: string): { frontmatter: Frontmatter; body: string } {
  const fmMatch = md.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!fmMatch) return { frontmatter: {}, body: md.trim() };
  const fmRaw = fmMatch[1];
  const body = md.slice(fmMatch[0].length).trim();
  const frontmatter: Frontmatter = {};
  fmRaw.split('\n').forEach(line => {
    const idx = line.indexOf(':');
    if (idx === -1) return;
    const key = line.slice(0, idx).trim();
    let value = line.slice(idx + 1).trim();
    // strip quotes
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    // handle arrays like [a, b] or comma-separated
    if (value.startsWith('[') && value.endsWith(']')) {
      const inner = value.slice(1, -1).trim();
      frontmatter[key] = inner ? inner.split(',').map(s => s.trim().replace(/^['"]|['"]$/g, '')) : [];
    } else if (value.includes(',') && key.toLowerCase() === 'tags') {
      frontmatter[key] = value.split(',').map(s => s.trim());
    } else {
      frontmatter[key] = value;
    }
  });
  return { frontmatter, body };
}

function extractTitle(md: string, fm: Frontmatter, fallback: string): string {
  if (typeof fm.title === 'string' && fm.title) return fm.title;
  const firstHeading = md.split('\n').find(l => l.trim().startsWith('#'));
  if (firstHeading) return firstHeading.replace(/^#+\s*/, '').trim();
  return fallback;
}

function normalizeDate(input?: string): string {
  if (!input) return new Date().toISOString().split('T')[0];
  // assume yyyy-mm-dd or iso; take date part
  const d = input.split('T')[0];
  return d;
}

export function loadNotes(): Note[] {
  const files = import.meta.glob('../posts/notes/*.md', { query: '?raw', eager: true });
  const notes: Note[] = Object.entries(files).map(([path, module]) => {
    // import.meta.glob with eager: true and query: '?raw' returns the module with default export as string
    // However, sometimes it returns an object with 'default' property.
    // Let's handle both cases.
    const raw = typeof module === 'string' ? module : (module as any).default;
    
    if (typeof raw !== 'string') {
        console.warn(`[loadNotes] Unexpected content type for ${path}:`, typeof raw);
        return null;
    }

    const { frontmatter, body } = parseFrontmatter(raw);
    const filename = path.split('/').pop() || 'note';
    const title = extractTitle(body, frontmatter, filename.replace(/\.md$/i, ''));
    const tags = Array.isArray(frontmatter.tags) ? (frontmatter.tags as string[]) : (typeof frontmatter.tags === 'string' && frontmatter.tags ? frontmatter.tags.split(',').map(s => s.trim()) : []);
    const category = typeof frontmatter.category === 'string' && frontmatter.category ? frontmatter.category : '随笔';
    const createdAt = normalizeDate((frontmatter.createdAt as string) || (frontmatter.date as string));
    return {
      id: path,
      title,
      content: body,
      tags,
      createdAt,
      category,
    };
  }).filter((note): note is Note => note !== null);
  return notes.sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
}

export function loadMoments(): LifeMoment[] {
  const files = import.meta.glob('../posts/lifes/*.md', { query: '?raw', eager: true });
  const moments: LifeMoment[] = Object.entries(files).map(([path, module]) => {
    const raw = typeof module === 'string' ? module : (module as any).default;
    
    if (typeof raw !== 'string') {
        console.warn(`[loadMoments] Unexpected content type for ${path}:`, typeof raw);
        return null;
    }

    const { frontmatter, body } = parseFrontmatter(raw);
    const description = typeof frontmatter.description === 'string' && frontmatter.description ? (frontmatter.description as string) : body.split('\n').find(Boolean) || '';
    const title = typeof frontmatter.title === 'string' && frontmatter.title ? frontmatter.title : undefined;
    const imageUrl = typeof frontmatter.imageUrl === 'string' && frontmatter.imageUrl ? (frontmatter.imageUrl as string) : undefined;
    const date = normalizeDate(frontmatter.date as string);
    const location = typeof frontmatter.location === 'string' ? (frontmatter.location as string) : undefined;
    const tags = Array.isArray(frontmatter.tags) ? (frontmatter.tags as string[]) : (typeof frontmatter.tags === 'string' && frontmatter.tags ? frontmatter.tags.split(',').map(s => s.trim()) : undefined);
    return {
      id: path,
      title,
      imageUrl,
      description,
      content: body,
      date,
      location,
      tags,
    } as LifeMoment;
  }).filter((moment): moment is LifeMoment => moment !== null);
  return moments.sort((a, b) => (a.date < b.date ? 1 : -1));
}



function parseFrontmatter(md: string): { frontmatter: Record<string, any>; body: string } {
  const fmMatch = md.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!fmMatch) return { frontmatter: {}, body: md.trim() };
  const fmRaw = fmMatch[1];
  const body = md.slice(fmMatch[0].length).trim();
  const frontmatter: Record<string, any> = {};
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

const sample2 = `---
description: "新年的第一天在公园散步，阳光很好"
location: "Null"
imageUrl: "https://picsum.photos/seed/random123/800/600"
date: 2026-01-01
---

生活需要被记录，这是一条通过 \`posts/lifes/*.md\` 自动读取的示例。`;

const result = parseFrontmatter(sample2);
console.log(JSON.stringify(result, null, 2));

function normalizeDate(input?: string): string {
  if (!input) return new Date().toISOString().split('T')[0];
  // assume yyyy-mm-dd or iso; take date part
  const d = input.split('T')[0];
  return d;
}

console.log('Normalized Date:', normalizeDate(result.frontmatter.date));

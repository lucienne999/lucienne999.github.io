// 处理飞书迁移 HTML 中的数学公式
//
// 背景：remark-math 只在 markdown 阶段扫描 `$...$` / `$$...$$` 分隔符，而飞书导出的
// 表格/引用块是以整段 raw HTML 的形式存在的，对 remark-math 来说是一个不可见的黑盒，
// 于是写在 <td>/<blockquote> 里的公式会原样输出成 `$$...$$` 文本（不会被 KaTeX 渲染）。
//
// 该插件在 hast（HTML）阶段运行，需插在 rehype-raw 之后、rehype-katex 之前：
// rehype-raw 已经把 raw HTML 解析成真正的元素节点，我们再扫描文本节点里的分隔符，
// 补上 remark-math 会生成的 `math-inline` / `math-display` 标记，交给 rehype-katex 渲染。

import { visit, SKIP } from 'unist-util-visit';
import type { Root, Element, Text, ElementContent } from 'hast';

// 已被 remark-math 处理过、或本身就是代码的节点直接跳过，避免重复处理与误伤。
const SKIP_TAGS = new Set(['code', 'pre']);

const hasMathClass = (node: Element): boolean => {
  const className = node.properties?.className;
  const classes = Array.isArray(className) ? className : [];
  return classes.some(
    (c) => c === 'math' || c === 'math-inline' || c === 'math-display',
  );
};

const makeMathSpan = (value: string, display: boolean): Element => ({
  type: 'element',
  tagName: 'span',
  properties: { className: [display ? 'math-display' : 'math-inline'] },
  children: [{ type: 'text', value }],
});

// 把一个文本节点按 `$$...$$`（块级）和 `$...$`（行内）拆分成
// 文本 + math span 交替的节点数组；没有公式时返回 null。
const splitTextNode = (value: string): ElementContent[] | null => {
  // 块级优先匹配，再匹配行内；两者都要求分隔符成对出现且中间非空。
  const re = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g;
  let match: RegExpExecArray | null;
  let lastIndex = 0;
  const out: ElementContent[] = [];

  while ((match = re.exec(value))) {
    if (match.index > lastIndex) {
      out.push({ type: 'text', value: value.slice(lastIndex, match.index) });
    }
    const display = match[1] !== undefined;
    const expr = (display ? match[1] : match[2]).trim();
    out.push(makeMathSpan(expr, display));
    lastIndex = re.lastIndex;
  }

  if (out.length === 0) return null;
  if (lastIndex < value.length) {
    out.push({ type: 'text', value: value.slice(lastIndex) });
  }
  return out;
};

export default function rehypeRawMath() {
  return (tree: Root) => {
    visit(tree, 'element', (node: Element) => {
      // 跳过代码块与已经被 remark-math 标记好的数学节点。
      if (SKIP_TAGS.has(node.tagName) || hasMathClass(node)) {
        return SKIP;
      }

      const children = node.children;
      if (!children || children.length === 0) return;

      const next: ElementContent[] = [];
      let changed = false;

      for (const child of children) {
        if (child.type === 'text' && (child as Text).value.includes('$')) {
          const replaced = splitTextNode((child as Text).value);
          if (replaced) {
            next.push(...replaced);
            changed = true;
            continue;
          }
        }
        next.push(child as ElementContent);
      }

      if (changed) {
        node.children = next;
      }
    });
  };
}

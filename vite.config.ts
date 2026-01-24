
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 注意：如果你的仓库名是 'my-site'，base 应该是 '/my-site/'
// 如果是用户名.github.io，则 base 应该是 '/'
export default defineConfig({
  plugins: [react()],
  base: './', // 使用相对路径可以兼容大多数 GitHub Pages 部署场景
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          utils: ['react-markdown', 'remark-gfm', 'remark-math', 'rehype-katex'],
        },
      },
    },
  }
});

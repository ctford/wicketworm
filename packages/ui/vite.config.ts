import { defineConfig } from 'vite';

export default defineConfig({
  base: '/wicketworm/',
  server: {
    port: 5173
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
});

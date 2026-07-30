import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5183,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8017', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8017', ws: true },
    },
  },
  test: {
    environment: 'jsdom',
  },
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    allowedHosts: ['.trycloudflare.com'],
  },
  preview: {
    host: true,
    allowedHosts: ['.trycloudflare.com'],
  },
  css: {
    postcss: './postcss.config.js',
  },
})

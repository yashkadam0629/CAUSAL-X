/// <reference types="vitest/config" />
import process from 'node:process'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { '/api': process.env.VITE_API_TARGET || 'http://127.0.0.1:8000' } },
  test: { environment: 'jsdom', globals: true, setupFiles: './src/test-setup.ts' },
})

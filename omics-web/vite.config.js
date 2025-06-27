import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
// import commonjs from 'vite-plugin-commonjs';
import path from 'path'
import tailwindcss from "@tailwindcss/vite";


export default defineConfig({
  plugins: [
    // commonjs(),
    tailwindcss(),
    svelte()
  ],
  server: {
    proxy: {
      '/api': {
        target: 'http://10.120.17.93:3538',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  resolve: {
    alias: {
      '@img': path.resolve(__dirname, './img')
    }
  }
})

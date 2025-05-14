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
  resolve: {
    alias: {
      '@img': path.resolve(__dirname, './img')
    }
  }
})

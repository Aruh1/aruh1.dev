// @ts-check
import { defineConfig } from "astro/config"
import { satteri } from "@astrojs/markdown-satteri"
import sitemap from "@astrojs/sitemap"
import tailwindcss from "@tailwindcss/vite"
import Icons from "unplugin-icons/vite"

// https://astro.build/config
export default defineConfig({
    site: "https://aruh1.dev",
    integrations: [sitemap()],
    markdown: {
        processor: satteri()
    },
    vite: {
        plugins: [
            tailwindcss(),
            Icons({
                compiler: "astro"
            })
        ]
    }
})

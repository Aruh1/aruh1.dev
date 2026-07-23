// @ts-check
import { defineConfig } from "astro/config"
import { satteri } from "@astrojs/markdown-satteri"
import tailwindcss from "@tailwindcss/vite"
import Icons from "unplugin-icons/vite"

// https://astro.build/config
export default defineConfig({
    site: "https://aruh1.dev",
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

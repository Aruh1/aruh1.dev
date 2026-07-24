// @ts-check
import { defineConfig } from "astro/config"
import { satteri } from "@astrojs/markdown-satteri"
import sitemap from "@astrojs/sitemap"
import tailwindcss from "@tailwindcss/vite"
import Icons from "unplugin-icons/vite"
import { shortlinkEntries } from "./src/data/shortlinks.ts"

const excludedPaths = new Set([
    "/404",
    "/404.html",
    ...shortlinkEntries.map(entry => `/${entry.slug}`),
    ...shortlinkEntries.map(entry => `/${entry.slug}/`)
])

// https://astro.build/config
export default defineConfig({
    site: "https://aruh1.dev",
    integrations: [
        sitemap({
            filter: page => {
                const pathname = new URL(page).pathname
                return !excludedPaths.has(pathname)
            }
        })
    ],
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

import type { APIRoute } from "astro"
import { site } from "../data/site"
import { renderOgImage } from "../lib/og"

export const GET: APIRoute = () =>
    renderOgImage({
        title: site.title,
        description: site.description,
        eyebrow: "Nerd developer",
        lang: site.language
    })

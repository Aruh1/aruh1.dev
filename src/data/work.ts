export const work = {
    heading: "MY WORK",
    intro: "Projects, tools and experiments I have built or keep improving.",
    work: [
        {
            id: "#001",
            status: "Live",
            type: "Web",
            title: "Yuramedia Link",
            href: "https://www.yuramedia.com",
            description: "A curated link hub for Yuramedia, featuring quick access to all official resources.",
            tags: ["Next.JS", "TypeScript", "PostgreSQL", "Tailwind CSS", "Shadcn/ui"]
        },
        {
            id: "#002",
            status: "Live",
            type: "Web",
            title: "Yuuume ASS Converter",
            href: "https://github.com/yuramedia/convert",
            description: "An ASS subtitle converter for Japanese subtitles.",
            tags: ["Astro", "TypeScript", "Shadcn/ui"]
        },
        {
            id: "#003",
            status: "Live",
            type: "Discord Bot",
            title: "Autoclaim Bot",
            href: "https://github.com/yuramedia/autoclaim-bot",
            description: "A Discord bot for autoclaim feature for your server",
            tags: ["Discord.JS", "TypeScript", "MongoDB"]
        },
        {
            id: "#004",
            status: "Live",
            type: "Web",
            title: "HSR Showcase",
            href: "https://github.com/Aruh1/hsr-showcase",
            description: "A web application for showcasing Honkai: Star Rail characters.",
            tags: ["Next.js", "TypeScript", "Tailwind CSS"]
        }
    ]
} as const

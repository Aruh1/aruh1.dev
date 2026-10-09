export type ExperienceVideo = {
    title: string
    url: string
    youtubeId: string
    badge?: string
}

export type ExperienceItem = {
    role: string
    company: string
    period: string
    location: string
    type: string
    description: string
    highlights: string[]
    skills: string[]
    href?: string
    videos?: ExperienceVideo[]
}

export const experience = {
    hero: {
        eyebrow: "/experience",
        titleLines: ["Career Journey", "& Background."],
        description: "A comprehensive overview of my professional experience, leadership roles, and technical skills."
    },
    experiences: [
        {
            role: "Chief Technology Officer",
            company: "Yuramedia Link",
            period: "Jan 2025 — Present",
            location: "Remote",
            type: "Freelance",
            description:
                "Leading technology management, full-stack development, and digital service infrastructure for localization services.",
            highlights: [
                "Managing technical strategy, web platform architecture, and full-stack development workflows.",
                "Overseeing technology operations and digital localization service pipelines."
            ],
            skills: ["Technology Management", "Full-Stack Development", "Web Architecture"]
        },
        {
            role: "Freelance Translator",
            company: "Yuramedia Link",
            period: "Feb 2021 — Sep 2025",
            location: "Remote",
            type: "Freelance",
            description:
                "Provided accurate and creative subtitle translations and quality assurance for films, videos, and multimedia content.",
            highlights: [
                "Provided accurate and creative translations for films, videos, and other multimedia materials.",
                "Ensured the quality and consistency of subtitle translations through editing and quality checking processes.",
                "Adapted to various film genres with a deep understanding of entertainment industry terminology.",
                "Integrated cultural and contextual understanding to convey messages accurately in translation."
            ],
            skills: ["WordPress", "SEO", "Localization", "Quality Checking"],
            videos: [
                {
                    title: "I Want to Love You Till Your Dying Day @crunchyroll",
                    url: "https://youtu.be/sivYQYsp2eM",
                    youtubeId: "sivYQYsp2eM",
                    badge: "Crunchyroll • Subtitle Indonesia"
                },
                {
                    title: "Roll Over and Die @crunchyroll",
                    url: "https://youtu.be/xeLkkDQR6P0",
                    youtubeId: "xeLkkDQR6P0",
                    badge: "Crunchyroll • Subtitle Indonesia"
                }
            ]
        },
        {
            role: "Co-Founder",
            company: "Lahza Pictura",
            period: "May 2024 — Present",
            location: "Malang, East Java, Indonesia",
            type: "Hybrid / Freelance",
            description:
                "Co-founded and managed visual media and photography projects, overseeing creative and operational workflows.",
            highlights: [
                "Co-leading business operations, client engagements, and visual production.",
                "Managing photography assets, creative direction, and brand presentation."
            ],
            skills: ["Photography", "Creative Direction", "Operations"]
        },
        {
            role: "Sales Promotions",
            company: "PT Pos Indonesia (Persero)",
            period: "May 2022 — Oct 2022",
            location: "Singosari, East Java, Indonesia",
            type: "Apprenticeship",
            description:
                "Completed a six-month apprenticeship at PT Pos Indonesia (Persero) in the Sales Promotions department as part of graduation requirements from SMK Negeri 2 Singosari.",
            highlights: [
                "Gained hands-on experience in promoting postal services and products to the public.",
                "Assisted in the development and execution of marketing campaigns and promotional events.",
                "Interacted directly with customers to explain services and enhance customer satisfaction."
            ],
            skills: ["Sales Promotion", "Marketing", "Customer Service", "Teamwork"]
        }
    ] satisfies ExperienceItem[]
} as const

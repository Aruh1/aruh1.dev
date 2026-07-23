export type Shortlink = {
    slug: string | readonly string[]
    href: string
}

export const shortlinks = [
    {
        slug: ["gh", "github"],
        href: "https://github.com/Aruh1"
    },
    {
        slug: ["discord", "dc"],
        href: "https://discord.com/users/454223855607742474"
    },
    {
        slug: ["email", "mail"],
        href: "mailto:hi@aruh1.dev"
    },
    {
        slug: ["donate", "sponsor", "ko-fi", "kofi"],
        href: "https://ko-fi.com/pololer"
    }
] as const satisfies readonly Shortlink[]

const toSlugList = (slug: Shortlink["slug"]) => (Array.isArray(slug) ? slug : [slug])

export const shortlinkEntries = shortlinks.flatMap(shortlink =>
    toSlugList(shortlink.slug).map(slug => ({
        slug,
        href: shortlink.href
    }))
)

export const shortlinkMap = new Map(shortlinkEntries.map(shortlink => [shortlink.slug, shortlink.href]))

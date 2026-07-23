export type IntroRole = {
    article: string
    label: string
}

export type CloneDirection = {
    x: string
    y: string
    top: string
    left: string
}

export const hero = {
    introPrefix: "This is Aruh1,",
    headline: "Nerd Developer",
    introRoles: [
        { article: "a", label: "backend developer" },
        { article: "a", label: "frontend developer" },
        { article: "a", label: "DevOps engineer" },
        { article: "a", label: "Fullstack Developer" }
    ] satisfies IntroRole[],
    cloneDirections: [
        { x: "0%", y: "100%", top: "-100%", left: "0%" },
        { x: "-100%", y: "0%", top: "0%", left: "100%" },
        { x: "0%", y: "-100%", top: "100%", left: "0%" },
        { x: "100%", y: "0%", top: "0%", left: "-100%" }
    ] satisfies CloneDirection[]
} as const

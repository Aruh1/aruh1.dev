import fs from "node:fs/promises"
import path from "node:path"
import sharp from "sharp"

const PUBLIC_DIR = path.resolve("public")

// Clean, modern SVG logo for "h1" with brand orange (#ff8b2b) accent
const svgContent = `<svg width="500" height="500" viewBox="0 0 500 500" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="500" height="500" rx="100" fill="#000000"/>
  <!-- 'h' glyph in white -->
  <path d="M 130 130 L 130 370" stroke="#FFFFFF" stroke-width="52" stroke-linecap="round"/>
  <path d="M 130 250 C 130 195 240 195 240 250 L 240 370" stroke="#FFFFFF" stroke-width="52" stroke-linecap="round" stroke-linejoin="round"/>
  <!-- '1' glyph in theme orange (#ff8b2b) -->
  <path d="M 310 230 L 355 190 L 355 370" stroke="#FF8B2B" stroke-width="52" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
`

async function generateFavicons() {
    console.log("Generating favicons for 'h1'...")

    // Save SVG
    const svgPath = path.join(PUBLIC_DIR, "favicon.svg")
    await fs.writeFile(svgPath, svgContent, "utf-8")
    console.log("Updated public/favicon.svg")

    const svgBuffer = Buffer.from(svgContent)

    // Sizes to generate
    const targets = [
        { name: "favicon-16x16.png", size: 16 },
        { name: "favicon-32x32.png", size: 32 },
        { name: "apple-touch-icon.png", size: 180 },
        { name: "android-chrome-192x192.png", size: 192 },
        { name: "android-chrome-512x512.png", size: 512 },
        { name: "favicon-orig.png", size: 500 }
    ]

    for (const target of targets) {
        const outputPath = path.join(PUBLIC_DIR, target.name)
        await sharp(svgBuffer).resize(target.size, target.size).png().toFile(outputPath)
        console.log(`Generated ${target.name} (${target.size}x${target.size})`)
    }

    // Generate multi-size favicon.ico (combining 16, 32, 48) using sharp PNG icon format
    const icoPath = path.join(PUBLIC_DIR, "favicon.ico")
    await sharp(svgBuffer).resize(32, 32).toFormat("png").toFile(icoPath)
    console.log("Updated public/favicon.ico")

    console.log("All favicons generated successfully!")
}

generateFavicons().catch(err => {
    console.error("Error generating favicons:", err)
    process.exit(1)
})

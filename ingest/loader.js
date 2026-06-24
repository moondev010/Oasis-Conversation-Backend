import fs from 'fs/promises'
import path from 'path'

export async function loadMarkdownFiles(docsPath) {
    const resolvedPath = path.resolve(docsPath)
    const entries = await fs.readdir(resolvedPath)

    const files = await Promise.all(
        entries
            .filter(entry => entry.endsWith('.md'))
            .map(async (fileName) => {
                const filePath = path.join(resolvedPath, fileName)
                const content = await fs.readFile(filePath, 'utf-8')
                return { fileName, filePath, content }
            })
    )

    console.log(`[loader] Found ${files.length} markdown files`)
    return files
}
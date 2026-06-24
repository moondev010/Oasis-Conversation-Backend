import matter from 'gray-matter'

export function parseMarkdownFiles(files) {
  const parsed = files.map(file => ({
    fileName: file.fileName,
    filePath: file.filePath,
    content: matter(file.content).content.trim()
  }))

  console.log(`[parser] Parsed ${parsed.length} markdown files`)
  return parsed
}
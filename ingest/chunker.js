import { encode, decode } from 'gpt-tokenizer'

function splitBySections(text) {
  const lines = text.split('\n')
  const sections = []
  let currentHeading = null
  let currentLines = []

  for (const line of lines) {
    if (line.match(/^#{1,6}\s+/)) {
      if (currentLines.length > 0) {
        sections.push({
          heading: currentHeading,
          text: currentLines.join('\n').trim()
        })
      }
      currentHeading = line.replace(/^#{1,6}\s+/, '').trim()
      currentLines = []
    } else {
      currentLines.push(line)
    }
  }

  if (currentLines.length > 0) {
    sections.push({
      heading: currentHeading,
      text: currentLines.join('\n').trim()
    })
  }

  return sections.filter(section => section.text.length > 0)
}

function chunkSection(section, chunkSize, chunkOverlap) {
  const tokens = encode(section.text)
  const chunks = []
  let start = 0

  while (start < tokens.length) {
    const end = Math.min(start + chunkSize, tokens.length)
    const chunkTokens = tokens.slice(start, end)
    const text = decode(chunkTokens).trim()

    chunks.push({
      heading: section.heading,
      text: section.heading ? `${section.heading}\n\n${text}` : text
    })

    start += chunkSize - chunkOverlap
  }

  return chunks
}

export function chunkParsedFiles(parsedFiles, chunkSize, chunkOverlap) {
  const chunks = []

  for (const file of parsedFiles) {
    const sections = splitBySections(file.content)

    sections.forEach(section => {
      const sectionChunks = chunkSection(section, chunkSize, chunkOverlap)

      sectionChunks.forEach((chunk, index) => {
        chunks.push({
          text: chunk.text,
          heading: chunk.heading,
          index,
          fileName: file.fileName
        })
      })
    })
  }

  console.log(`[chunker] Generated ${chunks.length} chunks`)
  return chunks
}
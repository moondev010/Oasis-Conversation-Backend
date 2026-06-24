import { loadMarkdownFiles } from './ingest/loader.js'
import { parseMarkdownFiles } from './ingest/parser.js'
import { chunkParsedFiles } from './ingest/chunker.js'
import { embedChunks } from './embedder.js'
import { initVectorStore, insertChunks } from './vectorstore.js'

const {
    DOCS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_EMBEDDING_MODEL,
    LANCEDB_PATH,
    LANCEDB_TABLE
} = process.env

console.log('[ingest] OLLAMA_HOST:', process.env.OLLAMA_HOST)
console.log('[ingest] OLLAMA_API_KEY:', process.env.OLLAMA_API_KEY ? 'loaded ✅' : 'missing ❌')

async function ingest() {
    console.log('[ingest] Starting ingestion pipeline...')

    const files = await loadMarkdownFiles(DOCS_PATH)
    const parsed = parseMarkdownFiles(files)
    const chunks = chunkParsedFiles(parsed, parseInt(CHUNK_SIZE), parseInt(CHUNK_OVERLAP))
    const embeddedChunks = await embedChunks(chunks, OLLAMA_EMBEDDING_MODEL)

    const dimensions = embeddedChunks[0].embedding.length
    await initVectorStore(LANCEDB_PATH, LANCEDB_TABLE, dimensions)
    await insertChunks(embeddedChunks)

    console.log('[ingest] Ingestion complete!')
}

ingest().catch(err => {
    console.error('[ingest] Error:', err)
    process.exit(1)
})
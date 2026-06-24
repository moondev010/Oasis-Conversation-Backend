import express from 'express'
import cors from 'cors'
import { initVectorStore } from './vectorstore.js'
import { runPipeline } from './pipeline.js'

const app = express()
app.use(cors())
app.use(express.json())

const {
  PORT,
  LANCEDB_PATH,
  LANCEDB_TABLE,
  OLLAMA_HOST,
  OLLAMA_API_KEY,
  OLLAMA_LLM_MODEL,
  OLLAMA_EMBEDDING_MODEL
} = process.env

const sessions = new Map()

app.post('/chat', async (req, res) => {
  try {
    if (!req.body || Object.keys(req.body).length === 0) {
      return res.status(400).json({ error: 'Request body is empty' })
    }

    const { message, sessionId } = req.body

    if (!message) {
      return res.status(400).json({ error: 'message field is required' })
    }

    if (typeof message !== 'string' || message.trim().length === 0) {
      return res.status(400).json({ error: 'message must be a non-empty string' })
    }

    if (!sessionId) {
      return res.status(400).json({ error: 'sessionId field is required' })
    }

    if (!sessions.has(sessionId)) {
      sessions.set(sessionId, [])
    }

    const history = sessions.get(sessionId)

    const response = await runPipeline(
      message.trim(),
      history,
      OLLAMA_LLM_MODEL,
      OLLAMA_EMBEDDING_MODEL,
      OLLAMA_HOST,
      OLLAMA_API_KEY
    )

    history.push({ role: 'user', content: message.trim() })
    history.push({ role: 'assistant', content: response })

    return res.json({ response })
  } catch (err) {
    console.error('[server] Error processing request:', err)
    return res.status(500).json({ error: 'Internal server error' })
  }
})

async function start () {
  await initVectorStore(LANCEDB_PATH, LANCEDB_TABLE)
  app.listen(PORT, () => {
    console.log(`[server] Oasis running on port ${PORT}`)
  })
}

start().catch(err => {
  console.error('[server] Failed to start:', err)
  process.exit(1)
})
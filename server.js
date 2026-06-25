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

const SESSION_TTL_MS = parseInt(process.env.SESSION_TTL_MS || String(30 * 60 * 1000))  // default 30 min
const SESSION_MAX = parseInt(process.env.SESSION_MAX || '100')

// Each entry: { history: [], lastActive: Date.now() }
const sessions = new Map()

function evictExpired() {
    const now = Date.now()
    for (const [id, session] of sessions) {
        if (now - session.lastActive > SESSION_TTL_MS) {
            sessions.delete(id)
            console.log(`[server] Session evicted (TTL): ${id}`)
        }
    }
}

function getOrCreateSession(sessionId) {
    evictExpired()

    if (sessions.has(sessionId)) {
        const session = sessions.get(sessionId)
        session.lastActive = Date.now()
        return session.history
    }

    // Evict oldest session if at cap
    if (sessions.size >= SESSION_MAX) {
        let oldestId = null
        let oldestTime = Infinity
        for (const [id, session] of sessions) {
            if (session.lastActive < oldestTime) {
                oldestTime = session.lastActive
                oldestId = id
            }
        }
        sessions.delete(oldestId)
        console.log(`[server] Session evicted (cap): ${oldestId}`)
    }

    sessions.set(sessionId, { history: [], lastActive: Date.now() })
    return sessions.get(sessionId).history
}

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

        const history = getOrCreateSession(sessionId)

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

async function start() {
    await initVectorStore(LANCEDB_PATH, LANCEDB_TABLE)
    app.listen(PORT, () => {
        console.log(`[server] Oasis running on port ${PORT}`)
    })
}

start().catch(err => {
    console.error('[server] Failed to start:', err)
    process.exit(1)
})
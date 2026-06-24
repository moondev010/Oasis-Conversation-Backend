import { Ollama } from 'ollama'
import { readFileSync } from 'fs'
import { embedChunks } from './embedder.js'
import { searchSimilarChunks } from './vectorstore.js'

const SYSTEM_PROMPT = readFileSync('./system-prompt.md', 'utf-8')
const MAX_HISTORY = parseInt(process.env.MAX_HISTORY || '10')

export async function runPipeline(userMessage, history, llmModel, embeddingModel, ollamaHost, ollamaApiKey) {
    const ollama = new Ollama({
        host: ollamaHost,
        headers: {
            Authorization: `Bearer ${ollamaApiKey}`
        }
    })

    const [queryEmbedded] = await embedChunks(
        [{ text: userMessage }],
        embeddingModel
    )

    const relevantChunks = await searchSimilarChunks(queryEmbedded.embedding, 3, 1.05)

    const trimmedHistory = history.slice(-MAX_HISTORY)

    if (relevantChunks.length === 0 && trimmedHistory.length === 0) {
        return 'Hola, soy Oasis. Estoy aquí para apoyarte con temas de estrés, ansiedad y burnout. ¿En qué puedo ayudarte hoy?'
    }

    const context = relevantChunks
        .map(chunk => chunk.text)
        .join('\n\n')

    const systemPromptWithContext = relevantChunks.length > 0
        ? `${SYSTEM_PROMPT}\n\n## Información Disponible\n\n${context}`
        : SYSTEM_PROMPT

    const response = await ollama.chat({
        model: llmModel,
        messages: [
            { role: 'system', content: systemPromptWithContext },
            ...trimmedHistory,
            { role: 'user', content: userMessage }
        ]
    })

    return response.message.content
}
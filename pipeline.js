import { Ollama } from 'ollama'
import { readFileSync } from 'fs'
import { embedChunks } from './embedder.js'
import { searchSimilarChunks } from './vectorstore.js'

const SYSTEM_PROMPT = readFileSync('./system-prompt.md', 'utf-8')
const MAX_HISTORY = parseInt(process.env.MAX_HISTORY || '10')

// Below this → in scope with context
const CONTEXT_THRESHOLD = parseFloat(process.env.CONTEXT_THRESHOLD || '0.5')
// Above this → off-topic, redirect
const SCOPE_THRESHOLD = parseFloat(process.env.SCOPE_THRESHOLD || '0.55')

/**
 * Streams the LLM response token by token.
 *
 * @param {string}   userMessage
 * @param {Array}    history        - Conversation history array (mutated by caller after stream ends)
 * @param {string}   llmModel
 * @param {string}   embeddingModel
 * @param {string}   ollamaHost
 * @param {string}   ollamaApiKey
 * @param {Function} onToken        - Called with each token string as it arrives
 * @returns {Promise<string>}       - Resolves with the full assembled response when streaming is done
 */
export async function streamPipeline(userMessage, history, llmModel, embeddingModel, ollamaHost, ollamaApiKey, onToken) {
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

    const relevantChunks = await searchSimilarChunks(queryEmbedded.embedding, 3, CONTEXT_THRESHOLD)
    const trimmedHistory = history.slice(-MAX_HISTORY)

    const isFirstMessage = trimmedHistory.length === 0
    const hasContext = relevantChunks.length > 0

    // Hardcoded greeting: stream it word-by-word so the client gets the same SSE flow
    if (isFirstMessage && !hasContext) {
        const greeting = 'Hola, soy Oasis. Estoy aquí para apoyarte con temas de bienestar emocional, estrés, ansiedad, burnout, sueño y vida universitaria. ¿En qué puedo ayudarte hoy?'
        for (const word of greeting.split(' ')) {
            onToken(word + ' ')
        }
        return greeting
    }

    // Use the best raw distance to decide scope, regardless of whether chunks passed the context threshold
    const bestDistance = hasContext
        ? relevantChunks[0]._distance
        : await getRawBestDistance(queryEmbedded.embedding)

    let systemPromptFinal

    if (hasContext) {
        // In scope with relevant context — inject it
        const context = relevantChunks.map(chunk => chunk.text).join('\n\n')
        systemPromptFinal = `${SYSTEM_PROMPT}\n\n## Información Disponible\n\n${context}`
    } else if (bestDistance <= SCOPE_THRESHOLD) {
        // In scope emotionally but no specific chunk matched — respond empathetically without inventing
        systemPromptFinal = `${SYSTEM_PROMPT}\n\n## Nota\nEl tema del estudiante está relacionado con su bienestar emocional o académico pero no se encontró información específica. Responde con empatía, valida su situación y sugiere que hable con un orientador o los servicios psicológicos de su universidad para recibir apoyo más específico. No ofrezcas pasos ni estrategias detalladas que no estén respaldadas por el contexto.`
    } else {
        // Off-topic — redirect
        systemPromptFinal = `${SYSTEM_PROMPT}\n\n## Nota\nEl mensaje del estudiante no está relacionado con salud mental o bienestar emocional. Indícale con amabilidad que solo puedes ayudarle con temas emocionales y psicológicos, y pregúntale cómo se ha sentido emocionalmente.`
    }

    const stream = await ollama.chat({
        model: llmModel,
        messages: [
            { role: 'system', content: systemPromptFinal },
            ...trimmedHistory,
            { role: 'user', content: userMessage }
        ],
        stream: true
    })

    let fullResponse = ''

    for await (const chunk of stream) {
        const token = chunk.message?.content ?? ''
        if (token) {
            onToken(token)
            fullResponse += token
        }
    }

    return fullResponse
}

// Fetches the best raw distance without a threshold filter
async function getRawBestDistance(queryEmbedding) {
    const results = await searchSimilarChunks(queryEmbedding, 1, Infinity)
    return results.length > 0 ? results[0]._distance : Infinity
}
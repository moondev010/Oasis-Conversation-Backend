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

    // ── STEP 3: DETERMINE THE EMOTIONAL SCOPE OF THE MESSAGE ─────────────────
    //
    // We need to know how "close" the user's message is to our knowledge base,
    // even if no chunk passed the strict CONTEXT_THRESHOLD filter.
    //
    // If we already have relevant chunks, the best (closest) distance is simply
    // the distance of the first result (results are sorted closest-first).
    //
    // If we have no chunks (none passed the filter), we do a separate search
    // with no distance limit at all (Infinity) just to get the raw best distance.
    // This tells us: "how far away is even the closest thing we know about?"

    const bestDistance = hasContext
        ? relevantChunks[0]._distance
        : await getRawBestDistance(queryEmbedded.embedding)

    // ── STEP 4: BUILD THE FINAL SYSTEM PROMPT ────────────────────────────────
    //
    // Based on the distance result, we decide which "version" of the system
    // prompt to send to the AI. This steers its behavior for this specific message.
    //
    // There are three possible situations:

    let systemPromptFinal

    if (hasContext) {
        // SITUATION A: We found relevant information in our database.
        // We join all the relevant text chunks into one block and append it
        // to the system prompt under "Información Disponible".
        // The AI will use this as its source of truth when answering.

        const context = relevantChunks.map(chunk => chunk.text).join('\n\n')
        systemPromptFinal = `${SYSTEM_PROMPT}\n\n## Información Disponible\n\n${context}`
    } else if (bestDistance <= SCOPE_THRESHOLD) {

        // SITUATION B: No specific document matched, but the message is still
        // emotionally close to our domain (mental health / student wellbeing).
        // We tell the AI to respond with empathy but without inventing advice,
        // and to suggest professional support.

        systemPromptFinal = `${SYSTEM_PROMPT}\n\n## Nota\nEl tema del estudiante está relacionado con su bienestar emocional o académico pero no se encontró información específica. Responde con empatía, valida su situación y sugiere que hable con un orientador o los servicios psicológicos de su universidad para recibir apoyo más específico. No ofrezcas pasos ni estrategias detalladas que no estén respaldadas por el contexto.`
    } else {

        // SITUATION C: The message is far from anything in our database — off-topic.
        // We tell the AI to kindly redirect the user back to mental health topics.

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

    // ── STEP 6: PROCESS THE STREAM TOKEN BY TOKEN ────────────────────────────
    //
    // "for await...of" is a special loop for asynchronous streams.
    // It waits for each piece (chunk) to arrive from the AI before continuing.
    //
    // Each chunk contains a small fragment of text — sometimes a full word,
    // sometimes just part of a word, sometimes punctuation. This is normal.
    //
    // We use optional chaining (?.) to safely access chunk.message.content —
    // if for any reason the property doesn't exist, it returns undefined
    // instead of throwing an error. The ?? '' means "use empty string if undefined".
    //
    // We accumulate every token into fullResponse so we can return the
    // complete text at the end (needed to save it into session history)

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


// ─────────────────────────────────────────────────────────────────────────────
// HELPER FUNCTION: getRawBestDistance
//
// This function searches the vector database with NO distance limit (Infinity),
// meaning it will always return the single closest result regardless of how
// far away it is.
//
// We use this only when no chunks passed the CONTEXT_THRESHOLD filter,
// to find out whether the user's message is at least in the general
// neighborhood of mental health topics (below SCOPE_THRESHOLD) or
// completely unrelated (above SCOPE_THRESHOLD).
// ────

// Fetches the best raw distance without a threshold filter
async function getRawBestDistance(queryEmbedding) {
    const results = await searchSimilarChunks(queryEmbedding, 1, Infinity)
    return results.length > 0 ? results[0]._distance : Infinity
}
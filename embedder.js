import ollama from 'ollama'

export async function embedChunks(chunks, model) {
    const response = await ollama.embed({
        model: model,
        input: chunks.map(chunk => chunk.text)
    })

    const embedded = chunks.map((chunk, index) => ({
        ...chunk,
        embedding: response.embeddings[index]
    }))

    console.log(`[embedder] Generated ${embedded.length} embeddings`)
    return embedded
}
import { connect } from '@lancedb/lancedb'

let db
let table

async function getDb(dbPath) {
    if (!db) {
        db = await connect(dbPath)
    }
    return db
}

export async function initVectorStore(dbPath, tableName, dimensions) {
    db = await getDb(dbPath)
    const tableNames = await db.tableNames()

    if (tableNames.includes(tableName)) {
        table = await db.openTable(tableName)
        console.log(`[vectorstore] Opened existing table "${tableName}"`)
    } else {
        table = await db.createTable(tableName, [
            {
                text: '',
                heading: '',
                fileName: '',
                index: 0,
                embedding: Array(dimensions).fill(0)
            }
        ])
        console.log(`[vectorstore] Created new table "${tableName}"`)
    }

    return table
}

export async function insertChunks(embeddedChunks) {
    await table.add(embeddedChunks)
    console.log(`[vectorstore] Inserted ${embeddedChunks.length} chunks`)
}

export async function searchSimilarChunks(queryEmbedding, topK = 5, similarityThreshold = 0.5) {
    const results = await table
        .vectorSearch(queryEmbedding)
        .limit(topK)
        .toArray()

    console.log('[vectorstore] Raw distance scores:')
    results.forEach((result, index) => {
        console.log(`  Chunk ${index + 1}: _distance=${result._distance.toFixed(4)} | heading="${result.heading}"`)
    })

    const filtered = results.filter(result => result._distance <= similarityThreshold)

    console.log(`[vectorstore] Found ${filtered.length}/${results.length} chunks below threshold (${similarityThreshold})`)
    return filtered
}
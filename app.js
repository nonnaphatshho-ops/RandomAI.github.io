// ============================================================
// MyPersonalAI - Browser AI
// ============================================================


// ============================================================
// ELEMENTS
// ============================================================

const input = document.getElementById("messageInput");
const button = document.getElementById("sendButton");
const chat = document.getElementById("chat");


// ============================================================
// CONFIG
// ============================================================

const MODEL_PATH = "./model/model.onnx";
const TOKENIZER_PATH = "./model/tokenizer.json";

const BLOCK_SIZE = 128;

const MAX_NEW_TOKENS = 80;

const TEMPERATURE = 0.8;


// ============================================================
// GLOBAL
// ============================================================

let session = null;
let tokenizer = null;

let modelReady = false;
let generating = false;


// ============================================================
// ADD MESSAGE
// ============================================================

function addMessage(type, text) {

    const message =
        document.createElement("div");

    message.className =
        `message ${type}`;


    const name =
        document.createElement("span");

    name.textContent =
        type === "user"
            ? "You:"
            : "AI:";


    const paragraph =
        document.createElement("p");

    paragraph.textContent =
        text;


    message.appendChild(name);
    message.appendChild(paragraph);


    chat.appendChild(message);


    chat.scrollTop =
        chat.scrollHeight;


    return paragraph;
}


// ============================================================
// LOAD TOKENIZER
// ============================================================

async function loadTokenizer() {

    console.log(
        "Loading tokenizer..."
    );


    const response =
        await fetch(
            TOKENIZER_PATH
        );


    if (!response.ok) {

        throw new Error(
            `Cannot load tokenizer.json: ${response.status}`
        );

    }


    const data =
        await response.json();


    tokenizer = {

        token_to_id:
            data.token_to_id,

        id_to_token: {}

    };


    for (
        const token in tokenizer.token_to_id
    ) {

        const id =
            Number(
                tokenizer.token_to_id[token]
            );


        tokenizer.id_to_token[id] =
            token;

    }


    console.log(
        "Tokenizer loaded."
    );


    console.log(
        "Vocabulary:",
        Object.keys(
            tokenizer.token_to_id
        ).length
    );

}


// ============================================================
// TOKENIZER
// ============================================================

function tokenize(text) {

    const pattern =
        /[\u0E01-\u0E2E][\u0E30-\u0E3A\u0E40-\u0E4E]*|[A-Za-z0-9_]+|[^\w\s]/gu;


    return text.match(pattern) || [];

}


// ============================================================
// ENCODE
// ============================================================

function encode(text) {

    const tokens =
        tokenize(text);


    const ids = [];


    for (
        const token of tokens
    ) {

        const id =
            tokenizer.token_to_id[token];


        if (
            id !== undefined
        ) {

            ids.push(
                Number(id)
            );

        } else {

            ids.push(
                Number(
                    tokenizer.token_to_id["<UNK>"]
                )
            );

        }

    }


    return ids;

}


// ============================================================
// DECODE
// ============================================================

function decode(ids) {

    const specialTokens =
        new Set([
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]);


    let text = "";


    for (
        const id of ids
    ) {

        const token =
            tokenizer.id_to_token[
                Number(id)
            ];


        if (!token) {

            continue;

        }


        if (
            specialTokens.has(token)
        ) {

            continue;

        }


        text += token;

    }


    return text;

}


// ============================================================
// SPECIAL TOKEN
// ============================================================

function getSpecialId(name) {

    const id =
        tokenizer.token_to_id[name];


    if (
        id === undefined
    ) {

        return -1;

    }


    return Number(id);

}


// ============================================================
// LOAD MODEL
// ============================================================

async function loadModel() {

    console.log(
        "Loading AI model..."
    );


    if (
        !window.ort
    ) {

        throw new Error(
            "ONNX Runtime Web was not loaded."
        );

    }


    console.log(
        "ONNX Runtime version:",
        window.ort.env
            ? "loaded"
            : "unknown"
    );


    // --------------------------------------------------------
    // Configure WASM
    // --------------------------------------------------------

    window.ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";


    // --------------------------------------------------------
    // Execution provider
    // --------------------------------------------------------

    let providers;


    if (
        "gpu" in navigator
    ) {

        console.log(
            "WebGPU detected."
        );


        providers = [
            "webgpu"
        ];

    } else {

        console.log(
            "WebGPU not available."
        );


        providers = [
            "wasm"
        ];

    }


    // --------------------------------------------------------
    // Create session
    // --------------------------------------------------------

    console.log(
        "Creating ONNX session..."
    );


    try {

        session =
            await window.ort.InferenceSession.create(
                MODEL_PATH,
                {
                    executionProviders:
                        providers,

                    graphOptimizationLevel:
                        "all"
                }
            );

    } catch (error) {

        console.warn(
            "WebGPU failed. Trying WASM..."
        );


        session =
            await window.ort.InferenceSession.create(
                MODEL_PATH,
                {
                    executionProviders:
                        ["wasm"],

                    graphOptimizationLevel:
                        "all"
                }
            );

    }


    console.log(
        "AI model loaded."
    );


    console.log(
        "Input names:",
        session.inputNames
    );


    console.log(
        "Output names:",
        session.outputNames
    );


    modelReady = true;

}


// ============================================================
// INITIALIZE
// ============================================================

async function initializeAI() {

    try {

        console.log(
            "================================="
        );

        console.log(
            "     MyPersonalAI starting..."
        );

        console.log(
            "================================="
        );


        await loadTokenizer();

        await loadModel();


        console.log(
            "================================="
        );

        console.log(
            "             AI READY"
        );

        console.log(
            "================================="
        );


        button.disabled =
            false;

        input.disabled =
            false;


        addMessage(
            "ai",
            "AI พร้อมแล้ว 🤖"
        );


    } catch (error) {

        console.error(
            "AI initialization failed:",
            error
        );


        addMessage(
            "ai",
            "โหลด AI ไม่สำเร็จ กรุณาดู Console (F12)"
        );

    }

}


// ============================================================
// SOFTMAX
// ============================================================

function softmax(
    logits,
    temperature
) {

    const values =
        Array.from(logits);


    const scaled =
        values.map(
            x =>
                x / temperature
        );


    const max =
        Math.max(
            ...scaled
        );


    const exps =
        scaled.map(
            x =>
                Math.exp(
                    x - max
                )
        );


    const sum =
        exps.reduce(
            (a, b) =>
                a + b,
            0
        );


    return exps.map(
        x =>
            x / sum
    );

}


// ============================================================
// SAMPLE TOKEN
// ============================================================

function sampleToken(logits) {

    const probabilities =
        softmax(
            logits,
            TEMPERATURE
        );


    let random =
        Math.random();


    for (
        let i = 0;
        i < probabilities.length;
        i++
    ) {

        random -=
            probabilities[i];


        if (
            random <= 0
        ) {

            return i;

        }

    }


    return (
        probabilities.length - 1
    );

}


// ============================================================
// GENERATE
// ============================================================

async function generate(prompt) {

    if (!modelReady) {

        throw new Error(
            "AI model is not ready."
        );

    }


    let inputIds =
        encode(prompt);


    if (
        inputIds.length === 0
    ) {

        return "";

    }


    // --------------------------------------------------------
    // Keep context inside model limit
    // --------------------------------------------------------

    if (
        inputIds.length >= BLOCK_SIZE
    ) {

        inputIds =
            inputIds.slice(
                -BLOCK_SIZE + 1
            );

    }


    const generatedIds = [];


    // --------------------------------------------------------
    // Generate
    // --------------------------------------------------------

    for (
        let step = 0;
        step < MAX_NEW_TOKENS;
        step++
    ) {

        let context =
            inputIds;


        if (
            context.length > BLOCK_SIZE
        ) {

            context =
                context.slice(
                    -BLOCK_SIZE
                );

        }


        // ----------------------------------------------------
        // Create padded input
        // ----------------------------------------------------

        const padded =
            new BigInt64Array(
                BLOCK_SIZE
            );


        const padId =
            getSpecialId(
                "<PAD>"
            );


        const actualPadId =
            padId >= 0
                ? padId
                : 0;


        for (
            let i = 0;
            i < BLOCK_SIZE;
            i++
        ) {

            padded[i] =
                BigInt(
                    actualPadId
                );

        }


        for (
            let i = 0;
            i < context.length;
            i++
        ) {

            padded[i] =
                BigInt(
                    context[i]
                );

        }


        // ----------------------------------------------------
        // Tensor
        // ----------------------------------------------------

        const tensor =
            new window.ort.Tensor(
                "int64",
                padded,
                [1, BLOCK_SIZE]
            );


        // ----------------------------------------------------
        // Run model
        // ----------------------------------------------------

        const results =
            await session.run({

                input_ids:
                    tensor

            });


        const outputName =
            session.outputNames[0];


        const output =
            results[
                outputName
            ];


        const vocabSize =
            output.dims[2];


        const position =
            context.length - 1;


        const start =
            position * vocabSize;


        const logits =
            output.data.slice(
                start,
                start + vocabSize
            );


        // ----------------------------------------------------
        // Select next token
        // ----------------------------------------------------

        const nextToken =
            sampleToken(
                logits
            );


        // ----------------------------------------------------
        // EOS
        // ----------------------------------------------------

        const eosId =
            getSpecialId(
                "<EOS>"
            );


        if (
            eosId >= 0 &&
            nextToken === eosId
        ) {

            break;

        }


        // ----------------------------------------------------
        // Add token
        // ----------------------------------------------------

        inputIds.push(
            nextToken
        );


        generatedIds.push(
            nextToken
        );


        // ----------------------------------------------------
        // Stop at next User message
        // ----------------------------------------------------

        const currentText =
            decode(
                generatedIds
            );


        if (
            currentText.includes(
                "User:"
            )
        ) {

            break;

        }


        // ให้ browser มีโอกาส render
        await new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    0
                )
        );

    }


    let response =
        decode(
            generatedIds
        );


    // --------------------------------------------------------
    // Clean
    // --------------------------------------------------------

    if (
        response.includes("User:")
    ) {

        response =
            response.split(
                "User:",
                1
            )[0];

    }


    response =
        response.replace(
            /AI:/g,
            ""
        );


    response =
        response.trim();


    return response;

}


// ============================================================
// SEND
// ============================================================

async function sendMessage() {

    if (generating) {

        return;

    }


    const text =
        input.value.trim();


    if (
        text === ""
    ) {

        return;

    }


    if (!modelReady) {

        addMessage(
            "ai",
            "AI กำลังโหลดอยู่ รอสักครู่นะ..."
        );

        return;

    }


    // --------------------------------------------------------
    // User message
    // --------------------------------------------------------

    addMessage(
        "user",
        text
    );


    input.value =
        "";


    generating =
        true;


    button.disabled =
        true;

    input.disabled =
        true;


    // --------------------------------------------------------
    // AI placeholder
    // --------------------------------------------------------

    const aiParagraph =
        addMessage(
            "ai",
            "กำลังคิด..."
        );


    try {

        const prompt =
            "User: " +
            text +
            "\nAI:";


        const response =
            await generate(
                prompt
            );


        if (
            response === ""
        ) {

            aiParagraph.textContent =
                "...";

        } else {

            aiParagraph.textContent =
                response;

        }


        chat.scrollTop =
            chat.scrollHeight;


    } catch (error) {

        console.error(
            "Generation error:",
            error
        );


        aiParagraph.textContent =
            "เกิดข้อผิดพลาดในการสร้างคำตอบ ดู Console (F12)";


    }


    generating =
        false;


    button.disabled =
        false;

    input.disabled =
        false;


    input.focus();

}


// ============================================================
// EVENTS
// ============================================================

button.addEventListener(
    "click",
    sendMessage
);


input.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter"
        ) {

            sendMessage();

        }

    }
);


// ============================================================
// START
// ============================================================

button.disabled =
    true;

input.disabled =
    true;


initializeAI();
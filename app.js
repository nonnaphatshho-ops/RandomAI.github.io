// ============================================================
// MyPersonalAI - Web Chat
// ============================================================

// ------------------------------------------------------------
// DOM
// ------------------------------------------------------------

const input = document.getElementById("messageInput");
const button = document.getElementById("sendButton");
const chat = document.getElementById("chat");


// ------------------------------------------------------------
// CONFIG
// ------------------------------------------------------------

const MODEL_PATH = "./model/model.onnx";
const TOKENIZER_PATH = "./model/tokenizer.json";

const BLOCK_SIZE = 128;

// จำนวนข้อความที่จำย้อนหลัง
// 10 messages = ประมาณ 5 รอบ User/AI
const MAX_HISTORY_MESSAGES = 10;

const MAX_NEW_TOKENS = 80;
const TEMPERATURE = 0.8;


// ------------------------------------------------------------
// GLOBAL
// ------------------------------------------------------------

let session = null;
let tokenizer = null;

let modelReady = false;
let generating = false;


// ------------------------------------------------------------
// CONVERSATION MEMORY
// ------------------------------------------------------------

// เก็บประวัติจริงของการสนทนา
//
// ตัวอย่าง:
//
// [
//     { role: "User", text: "สวัสดี" },
//     { role: "AI", text: "สวัสดีครับ" },
//     { role: "User", text: "นายชื่ออะไร" },
//     { role: "AI", text: "ผมชื่อ MyPersonalAI" }
// ]

let conversationHistory = [];


// ------------------------------------------------------------
// ADD MESSAGE TO CHAT UI
// ------------------------------------------------------------

function addMessage(type, text) {

    const message = document.createElement("div");

    message.className = "message " + type;

    const label = document.createElement("span");

    label.textContent =
        type === "user"
            ? "You:"
            : "AI:";

    const paragraph = document.createElement("p");

    paragraph.textContent = text;

    message.appendChild(label);
    message.appendChild(paragraph);

    chat.appendChild(message);

    chat.scrollTop = chat.scrollHeight;

    return paragraph;
}


// ------------------------------------------------------------
// CONVERSATION HISTORY
// ------------------------------------------------------------

function addToHistory(role, text) {

    conversationHistory.push({
        role: role,
        text: text
    });

    // จำกัดจำนวนข้อความ
    if (conversationHistory.length > MAX_HISTORY_MESSAGES) {

        conversationHistory =
            conversationHistory.slice(
                -MAX_HISTORY_MESSAGES
            );
    }
}


// ------------------------------------------------------------
// REMOVE LAST HISTORY MESSAGE
// ------------------------------------------------------------

function removeLastHistoryMessage() {

    if (conversationHistory.length > 0) {

        conversationHistory.pop();
    }
}


// ------------------------------------------------------------
// BUILD PROMPT
// ------------------------------------------------------------

function buildPrompt() {

    let prompt = "";

    for (const message of conversationHistory) {

        if (message.role === "User") {

            prompt +=
                "User: " +
                message.text +
                "\n";

        } else if (message.role === "AI") {

            prompt +=
                "AI: " +
                message.text +
                "\n";
        }
    }

    // ตอนสุดท้ายต้องให้โมเดลตอบ AI
    prompt += "AI:";

    return prompt;
}


// ------------------------------------------------------------
// DEBUG CONTEXT
// ------------------------------------------------------------

function debugContext(prompt) {

    console.log("");
    console.log("========================================");
    console.log("MY PERSONAL AI - CONTEXT");
    console.log("========================================");

    console.log(
        "Messages:",
        conversationHistory.length,
        "/",
        MAX_HISTORY_MESSAGES
    );

    console.log("");

    conversationHistory.forEach(
        (message, index) => {

            console.log(
                `${index + 1}. ${message.role}:`,
                message.text
            );
        }
    );

    console.log("");

    const tokenCount = encode(prompt).length;

    console.log(
        "Prompt tokens:",
        tokenCount,
        "/",
        BLOCK_SIZE
    );

    console.log("");

    console.log("Prompt sent to model:");

    console.log("----------------------------------------");

    console.log(prompt);

    console.log("----------------------------------------");

    console.log("========================================");
    console.log("");
}


// ------------------------------------------------------------
// TOKENIZER
// ------------------------------------------------------------

async function loadTokenizer() {

    const response =
        await fetch(TOKENIZER_PATH);

    if (!response.ok) {

        throw new Error(
            "Cannot load tokenizer.json"
        );
    }

    const data =
        await response.json();

    tokenizer = {

        token_to_id:
            data.token_to_id || {},

        id_to_token: {}
    };


    // สร้าง reverse dictionary
    for (
        const [token, id]
        of Object.entries(tokenizer.token_to_id)
    ) {

        tokenizer.id_to_token[id] = token;
    }


    console.log(
        "Tokenizer loaded:",
        Object.keys(
            tokenizer.token_to_id
        ).length,
        "tokens"
    );
}


// ------------------------------------------------------------
// TOKENIZE
// ------------------------------------------------------------

function tokenize(text) {

    return text.match(
        /[\u0E01-\u0E2E][\u0E30-\u0E3A\u0E40-\u0E4E]*|[A-Za-z0-9_]+|[^\w\s]/gu
    ) || [];
}


// ------------------------------------------------------------
// ENCODE
// ------------------------------------------------------------

function encode(text) {

    const tokens =
        tokenize(text);

    const unkId =
        getSpecialId("<UNK>");

    return tokens.map(
        token => {

            if (
                tokenizer.token_to_id[token]
                !== undefined
            ) {

                return tokenizer.token_to_id[token];
            }

            return unkId;
        }
    );
}


// ------------------------------------------------------------
// DECODE
// ------------------------------------------------------------

function decode(ids) {

    const ignoredTokens = new Set([
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>"
    ]);

    let result = "";

    for (const id of ids) {

        const token =
            tokenizer.id_to_token[id];

        if (!token) {
            continue;
        }

        if (
            ignoredTokens.has(token)
        ) {
            continue;
        }

        result += token;
    }

    return result;
}


// ------------------------------------------------------------
// SPECIAL TOKEN
// ------------------------------------------------------------

function getSpecialId(name) {

    if (
        tokenizer &&
        tokenizer.token_to_id &&
        tokenizer.token_to_id[name]
        !== undefined
    ) {

        return tokenizer.token_to_id[name];
    }

    return -1;
}


// ------------------------------------------------------------
// LOAD MODEL
// ------------------------------------------------------------

async function loadModel() {

    console.log(
        "Loading ONNX model..."
    );

    try {

        // พยายามใช้ WebGPU ก่อน
        if ("gpu" in navigator) {

            console.log(
                "Trying WebGPU..."
            );

            try {

                session =
                    await ort.InferenceSession.create(
                        MODEL_PATH,
                        {
                            executionProviders: [
                                "webgpu"
                            ]
                        }
                    );

                console.log(
                    "Using WebGPU"
                );

            } catch (error) {

                console.warn(
                    "WebGPU failed, using WASM",
                    error
                );

                session =
                    await ort.InferenceSession.create(
                        MODEL_PATH,
                        {
                            executionProviders: [
                                "wasm"
                            ]
                        }
                    );
            }

        } else {

            console.log(
                "WebGPU unavailable, using WASM"
            );

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "wasm"
                        ]
                    }
                );
        }


        console.log(
            "Model inputs:",
            session.inputNames
        );

        console.log(
            "Model outputs:",
            session.outputNames
        );

        modelReady = true;

        console.log(
            "Model ready."
        );

    } catch (error) {

        console.error(
            "Model loading failed:",
            error
        );

        throw error;
    }
}


// ------------------------------------------------------------
// INITIALIZE AI
// ------------------------------------------------------------

async function initializeAI() {

    try {

        input.disabled = true;
        button.disabled = true;

        await loadTokenizer();

        await loadModel();

        input.disabled = false;
        button.disabled = false;

        addMessage(
            "ai",
            "AI พร้อมแล้ว 🤖"
        );

        input.focus();

    } catch (error) {

        console.error(error);

        addMessage(
            "ai",
            "โหลด AI ไม่สำเร็จ"
        );
    }
}


// ------------------------------------------------------------
// SOFTMAX
// ------------------------------------------------------------

function softmax(logits, temperature) {

    const scaled =
        logits.map(
            x => x / temperature
        );


    const maxLogit =
        Math.max(...scaled);


    const exps =
        scaled.map(
            x => Math.exp(x - maxLogit)
        );


    const sum =
        exps.reduce(
            (a, b) => a + b,
            0
        );


    return exps.map(
        x => x / sum
    );
}


// ------------------------------------------------------------
// SAMPLE TOKEN
// ------------------------------------------------------------

function sampleToken(logits) {

    const probabilities =
        softmax(
            logits,
            TEMPERATURE
        );


    const random =
        Math.random();


    let cumulative = 0;


    for (
        let i = 0;
        i < probabilities.length;
        i++
    ) {

        cumulative +=
            probabilities[i];


        if (
            random <= cumulative
        ) {

            return i;
        }
    }


    return probabilities.length - 1;
}


// ------------------------------------------------------------
// GENERATE
// ------------------------------------------------------------

async function generate(prompt) {

    let inputIds =
        encode(prompt);


    // --------------------------------------------------------
    // จำกัด context ตาม BLOCK_SIZE
    // --------------------------------------------------------

    if (
        inputIds.length >= BLOCK_SIZE
    ) {

        inputIds =
            inputIds.slice(
                -(BLOCK_SIZE - 1)
            );
    }


    const generatedIds = [];


    const padId =
        getSpecialId("<PAD>");

    const eosId =
        getSpecialId("<EOS>");


    for (
        let step = 0;
        step < MAX_NEW_TOKENS;
        step++
    ) {

        // ----------------------------------------------------
        // สร้าง input tensor
        // ----------------------------------------------------

        const inputArray =
            new BigInt64Array(
                BLOCK_SIZE
            );


        // เติม PAD
        for (
            let i = 0;
            i < BLOCK_SIZE;
            i++
        ) {

            inputArray[i] =
                BigInt(
                    padId >= 0
                        ? padId
                        : 0
                );
        }


        // ----------------------------------------------------
        // ใส่ context
        // ----------------------------------------------------

        for (
            let i = 0;
            i < inputIds.length;
            i++
        ) {

            inputArray[i] =
                BigInt(
                    inputIds[i]
                );
        }


        const tensor =
            new ort.Tensor(
                "int64",
                inputArray,
                [1, BLOCK_SIZE]
            );


        // ----------------------------------------------------
        // RUN MODEL
        // ----------------------------------------------------

        const outputs =
            await session.run({
                input_ids: tensor
            });


        const outputName =
            session.outputNames[0];


        const output =
            outputs[outputName];


        const dims =
            output.dims;


        // ----------------------------------------------------
        // vocab size
        // ----------------------------------------------------

        const vocabSize =
            dims[dims.length - 1];


        // ----------------------------------------------------
        // position ของ token ล่าสุด
        // ----------------------------------------------------

        const position =
            inputIds.length - 1;


        const start =
            position * vocabSize;


        const logits = [];


        for (
            let i = 0;
            i < vocabSize;
            i++
        ) {

            logits.push(
                output.data[start + i]
            );
        }


        // ----------------------------------------------------
        // เลือก token ใหม่
        // ----------------------------------------------------

        const nextToken =
            sampleToken(logits);


        // ----------------------------------------------------
        // EOS
        // ----------------------------------------------------

        if (
            eosId >= 0 &&
            nextToken === eosId
        ) {

            break;
        }


        // ----------------------------------------------------
        // เพิ่ม token
        // ----------------------------------------------------

        generatedIds.push(
            nextToken
        );

        inputIds.push(
            nextToken
        );


        // ----------------------------------------------------
        // หยุดถ้าโมเดลเริ่มสร้าง User ใหม่
        // ----------------------------------------------------

        const currentText =
            decode(generatedIds);


        if (
            currentText.includes("User:")
        ) {

            break;
        }


        // ----------------------------------------------------
        // ให้ browser หายใจ
        // ----------------------------------------------------

        await new Promise(
            resolve =>
                setTimeout(resolve, 0)
        );
    }


    // --------------------------------------------------------
    // Decode
    // --------------------------------------------------------

    let response =
        decode(generatedIds);


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    response =
        response
            .replace(/User:/g, "")
            .replace(/AI:/g, "")
            .trim();


    return response;
}


// ------------------------------------------------------------
// SEND MESSAGE
// ------------------------------------------------------------

async function sendMessage() {

    // ป้องกันกดซ้ำ
    if (generating) {
        return;
    }


    const text =
        input.value.trim();


    if (!text) {
        return;
    }


    if (!modelReady) {

        addMessage(
            "ai",
            "AI ยังโหลดไม่เสร็จ"
        );

        return;
    }


    // --------------------------------------------------------
    // USER MESSAGE
    // --------------------------------------------------------

    addMessage(
        "user",
        text
    );


    // --------------------------------------------------------
    // เพิ่ม User เข้า memory
    // --------------------------------------------------------

    addToHistory(
        "User",
        text
    );


    input.value = "";


    // --------------------------------------------------------
    // AI PLACEHOLDER
    // --------------------------------------------------------

    const aiParagraph =
        addMessage(
            "ai",
            "กำลังคิด..."
        );


    generating = true;

    input.disabled = true;
    button.disabled = true;


    try {

        // ----------------------------------------------------
        // สร้าง prompt จากประวัติทั้งหมด
        // ----------------------------------------------------

        const prompt =
            buildPrompt();


        // ----------------------------------------------------
        // DEBUG
        // ----------------------------------------------------

        debugContext(prompt);


        // ----------------------------------------------------
        // GENERATE
        // ----------------------------------------------------

        const response =
            await generate(prompt);


        // ----------------------------------------------------
        // แสดงผล
        // ----------------------------------------------------

        const finalResponse =
            response ||
            "...";


        aiParagraph.textContent =
            finalResponse;


        // ----------------------------------------------------
        // สำคัญ:
        // เพิ่มคำตอบ AI เข้า context หลัง generate เสร็จ
        // ----------------------------------------------------

        addToHistory(
            "AI",
            finalResponse
        );


        // ----------------------------------------------------
        // DEBUG หลังเพิ่ม AI
        // ----------------------------------------------------

        console.log(
            "AI response:",
            finalResponse
        );


        console.log(
            "Conversation memory:",
            conversationHistory
        );


    } catch (error) {

        console.error(
            "Generation error:",
            error
        );


        aiParagraph.textContent =
            "เกิดข้อผิดพลาดในการสร้างคำตอบ";


        // ----------------------------------------------------
        // ถ้า generate ไม่สำเร็จ
        // เอา User ล่าสุดออกจาก memory
        // ----------------------------------------------------

        removeLastHistoryMessage();
    }


    generating = false;

    input.disabled = false;
    button.disabled = false;

    input.focus();
}


// ------------------------------------------------------------
// CLEAR CONVERSATION
// ------------------------------------------------------------

function clearConversation() {

    conversationHistory = [];


    console.log(
        "Conversation history cleared."
    );
}


// เปิดให้เรียกจาก Browser Console ได้
window.clearConversation =
    clearConversation;


// ------------------------------------------------------------
// DEBUG MEMORY
// ------------------------------------------------------------

function showContext() {

    console.log("");
    console.log(
        "========== CURRENT MEMORY =========="
    );


    console.log(
        "Messages:",
        conversationHistory.length,
        "/",
        MAX_HISTORY_MESSAGES
    );


    conversationHistory.forEach(
        (message, index) => {

            console.log(
                `${index + 1}. ${message.role}: ${message.text}`
            );
        }
    );


    console.log(
        "===================================="
    );

    console.log("");
}


// เปิดให้เรียกจาก Browser Console ได้
window.showContext =
    showContext;


// ------------------------------------------------------------
// EVENTS
// ------------------------------------------------------------

button.addEventListener(
    "click",
    sendMessage
);


input.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }
    }
);


// ------------------------------------------------------------
// START
// ------------------------------------------------------------

input.disabled = true;
button.disabled = true;

initializeAI();

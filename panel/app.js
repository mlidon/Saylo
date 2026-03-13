const chat = document.getElementById("chat-container");
const textInput = document.getElementById("textInput");
const sendTextBtn = document.getElementById("sendText");
const audioPlayer = document.getElementById("audioPlayer");
const micBtn = document.getElementById("micBtn");
const typingIndicator = document.getElementById("typing-indicator");
const avatar = document.getElementById("avatar");
const API_URL = "http://localhost:8000";

let mediaRecorder;
let chunks = [];


function addMessage(sender, text) {
    const div = document.createElement("div");
    div.className = `message ${sender}`;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

// --- Button Send text  --- //
sendTextBtn.onclick = async () => {
    const text = textInput.value.trim();
    if (!text) return;

    addMessage("user", text);
    textInput.value = "";

    // Mostrar “pensando…”
    showThinking();
    avatarThinking();   // ← AÑADIDO AQUÍ

    const res = await fetch(`${API_URL}/api/text`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text })
    });

    const data = await res.json();

    // Ocultar “pensando…” cuando llega la respuesta
    hideThinking();

    addMessage("assistant", data.reply);

    // Avatar hablando
    avatarSpeaking();   // ← AÑADIDO AQUÍ

    audioPlayer.src = `${API_URL}${data.audio}`;
    audioPlayer.play();
    
    // Cuando termine el audio → avatar idle
    audioPlayer.onended = () => avatarIdle();
    forceAvatarIdle();
};


// --- Button on Click --- //
micBtn.onclick = async () => {
    // Si no está grabando, empezamos
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        chunks = [];

        mediaRecorder.ondataavailable = e => chunks.push(e.data);

        mediaRecorder.onstop = async () => {
            addMessage("system", "⏳ Procesando audio…");
            showThinking();
            avatarThinking();   // ← AÑADIDO AQUÍ

            const blob = new Blob(chunks, { type: "audio/wav" });
            chunks = [];

            const formData = new FormData();
            formData.append("file", blob, "audio.wav");

            const res = await fetch(`${API_URL}/api/audio`, {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            hideThinking();

            addMessage("user", data.transcript);
            addMessage("assistant", data.reply);

            // Avatar hablando
            avatarSpeaking();   // ← AÑADIDO AQUÍ

            audioPlayer.src = `${API_URL}${data.audio}`;
            audioPlayer.play();

            // Cuando termine el audio → avatar idle
            audioPlayer.onended = () => avatarIdle();   // ← AÑADIDO AQUÍ

            micBtn.classList.remove("recording");
            micBtn.innerHTML = `<span class="material-icons">mic</span>`;
        };


        mediaRecorder.start();

        // Indicador: grabando
        micBtn.classList.add("recording");
        micBtn.innerHTML = `<span class="material-icons">stop</span>`;
        addMessage("system", "🎙️ Grabando… pulsa de nuevo para parar.");
    } else {
        // Si está grabando, paramos
        mediaRecorder.stop();
    }
};


// --- Typing --- //
function showThinking() {
    typingIndicator.classList.remove("hidden");
}

function hideThinking() {
    typingIndicator.classList.add("hidden");
}



//--- AVATAR ---//
function avatarThinking() {
    avatar.classList.add("thinking");
    avatar.classList.remove("speaking");
}

function avatarSpeaking() {
    avatar.classList.add("speaking");
    avatar.classList.remove("thinking");
}

function avatarIdle() {
    avatar.classList.remove("thinking");
    avatar.classList.remove("speaking");
}

function forceAvatarIdle() {
    setTimeout(() => {
        avatarIdle();
    }, 3000); // 3 segundos después de empezar a hablar
}
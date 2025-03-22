package main

import (
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "strings"
    "sync"
    "time"

    "github.com/google/uuid"
    "github.com/sashabaranov/go-openai"
)

type Session struct {
    History []openai.ChatCompletionMessage
}

var (
    sessions = make(map[string]*Session)
    mutex    = &sync.Mutex{}
)

// getGrokResponse deve essere definita prima di essere chiamata nel main
func getGrokResponse(grokKey string, client *http.Client, question string) (string, error) {
    if grokKey == "" {
        return "", fmt.Errorf("GROK_API_KEY non è impostata")
    }
    // Sanifica il prompt per evitare problemi con caratteri speciali
    question = strings.ReplaceAll(question, "\n", " ")
    question = strings.ReplaceAll(question, "\"", "\\\"")
    // Usa un limite hardcoded per il log
    logLimit := 100
    if len(question) < logLimit {
        logLimit = len(question)
    }
    fmt.Println("Invio richiesta a Grok con prompt (prime 100 chars):", question[:logLimit], "...")

    // Aggiungi un meccanismo di retry
    var grokResp *http.Response
    var err error
    for attempt := 1; attempt <= 3; attempt++ {
        grokReq, err := http.NewRequest("POST", "https://api.x.ai/v1/chat/completions",
            strings.NewReader(fmt.Sprintf(`{"model":"grok","messages":[{"role":"user","content":"%s"}]}`, question)))
        if err != nil {
            return "", fmt.Errorf("errore nella creazione della richiesta a Grok (tentativo %d): %v", attempt, err)
        }
        grokReq.Header.Set("Authorization", "Bearer "+grokKey)
        grokReq.Header.Set("Content-Type", "application/json")
        grokResp, err = client.Do(grokReq)
        if err == nil {
            break
        }
        fmt.Printf("Errore con Grok (tentativo %d): %v\n", attempt, err)
        time.Sleep(time.Second * time.Duration(attempt)) // Attendi prima di riprovare
    }
    if err != nil {
        return "", fmt.Errorf("errore con Grok dopo 3 tentativi: %v", err)
    }
    defer grokResp.Body.Close()

    body, err := io.ReadAll(grokResp.Body)
    if err != nil {
        return "", fmt.Errorf("errore nella lettura della risposta di Grok: %v", err)
    }
    fmt.Println("Risposta grezza da Grok (status %d): %s", grokResp.StatusCode, string(body))

    var grokResult struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
        Error *struct {
            Message string `json:"message"`
        } `json:"error"`
    }
    if err := json.Unmarshal(body, &grokResult); err != nil {
        return "", fmt.Errorf("errore nel parsing di Grok: %v, risposta grezza: %s", err, string(body))
    }
    if grokResult.Error != nil {
        return "", fmt.Errorf("errore da xAI: %s", grokResult.Error.Message)
    }
    if len(grokResult.Choices) == 0 {
        return "", fmt.Errorf("nessuna risposta valida da Grok: %s", string(body))
    }
    return grokResult.Choices[0].Message.Content, nil
}

func main() {
    // Carica le chiavi API dalle variabili d'ambiente
    openAIKey := os.Getenv("OPENAI_API_KEY")
    deepSeekKey := os.Getenv("DEEPSEEK_API_KEY")
    geminiKey := os.Getenv("GEMINI_API_KEY")
    grokKey := "xai-SUGB15xF6kpFtSXVz73yAUjVeoQVUGM1VLFmX6Bj5HsN8hsSscWbs4wRw6YuXVAG0U4rbnheWgJ8tai0" // Chiave API di xAI

    fmt.Printf("Caricamento chiavi API...\n")
    if openAIKey == "" {
        fmt.Println("Errore: OPENAI_API_KEY non è impostata")
    } else {
        fmt.Println("OPENAI_API_KEY caricata correttamente")
    }
    if deepSeekKey == "" {
        fmt.Println("Errore: DEEPSEEK_API_KEY non è impostata")
    } else {
        fmt.Println("DEEPSEEK_API_KEY caricata correttamente")
    }
    if geminiKey == "" {
        fmt.Println("Errore: GEMINI_API_KEY non è impostata")
    } else {
        fmt.Println("GEMINI_API_KEY caricata correttamente")
    }
    if grokKey == "" {
        fmt.Println("Errore: GROK_API_KEY non è impostata")
    } else {
        fmt.Println("GROK_API_KEY caricata correttamente")
    }

    port := os.Getenv("PORT")
    if port == "" {
        fmt.Println("PORT non specificata, uso default :10000")
        port = "10000" // Default per test locali
    }

    openAIClient := openai.NewClient(openAIKey)
    client := &http.Client{
        Timeout: 20 * time.Second,
    }

    // Test preliminare per la chiave API di xAI
    fmt.Println("Test preliminare per la chiave API di xAI...")
    testPrompt := "Test: rispondi con 'OK' se la chiave API funziona."
    testResponse, err := getGrokResponse(grokKey, client, testPrompt)
    if err != nil {
        fmt.Printf("Errore nel test preliminare della chiave API di xAI: %v\n", err)
        fmt.Println("Procedo senza sintesi di Grok, usando un fallback.")
    } else {
        fmt.Println("Test preliminare riuscito. Risposta di Grok:", testResponse)
    }

    getDeepSeekResponse := func(messages []openai.ChatCompletionMessage) (string, error) {
        if deepSeekKey == "" {
            return "", fmt.Errorf("DEEPSEEK_API_KEY non è impostata")
        }
        var deepSeekMessages []map[string]string
        for _, msg := range messages {
            deepSeekMessages = append(deepSeekMessages, map[string]string{
                "role":    msg.Role,
                "content": msg.Content,
            })
        }
        body, err := json.Marshal(map[string]interface{}{
            "model":    "deepseek-chat",
            "messages": deepSeekMessages,
        })
        if err != nil {
            return "", fmt.Errorf("errore nella creazione del body JSON: %v", err)
        }
        req, err := http.NewRequest("POST", "https://api.deepseek.com/v1/chat/completions", strings.NewReader(string(body)))
        if err != nil {
            return "", fmt.Errorf("errore nella creazione della richiesta: %v", err)
        }
        req.Header.Set("Authorization", "Bearer "+deepSeekKey)
        req.Header.Set("Content-Type", "application/json")
        resp, err := client.Do(req)
        if err != nil {
            return "", fmt.Errorf("errore nella richiesta a DeepSeek: %v", err)
        }
        defer resp.Body.Close()
        bodyResp, err := io.ReadAll(resp.Body)
        if err != nil {
            return "", fmt.Errorf("errore nella lettura della risposta: %v", err)
        }
        if resp.StatusCode != http.StatusOK {
            return "", fmt.Errorf("risposta non valida da DeepSeek (status %d): %s", resp.StatusCode, string(bodyResp))
        }
        var result struct {
            Choices []struct {
                Message struct {
                    Content string `json:"content"`
                } `json:"message"`
            } `json:"choices"`
        }
        if err := json.Unmarshal(bodyResp, &result); err != nil {
            return "", fmt.Errorf("errore nel parsing JSON: %v, risposta grezza: %s", err, string(bodyResp))
        }
        if len(result.Choices) > 0 {
            return result.Choices[0].Message.Content, nil
        }
        return "", fmt.Errorf("nessuna risposta valida da DeepSeek: %s", string(bodyResp))
    }

    // Health check endpoint
    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        fmt.Println("Ricevuta richiesta su /health")
        w.WriteHeader(http.StatusOK)
        fmt.Fprintf(w, "Server is running on port %s", port)
    })

    // Serve la pagina HTML
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        // Commentato il reindirizzamento finché il dominio non è configurato
        // if r.Host == "arca-b-chat-ai.onrender.com" {
        //     http.Redirect(w, r, "https://arcabchat.com"+r.RequestURI, http.StatusMovedPermanently)
        //     return
        // }
        fmt.Println("Ricevuta richiesta su /")
        sessionID, err := r.Cookie("session_id")
        if err != nil || sessionID == nil {
            sessionID = &http.Cookie{
                Name:  "session_id",
                Value: uuid.New().String(),
                Path:  "/",
            }
            http.SetCookie(w, sessionID)
        }
        w.Header().Set("Content-Type", "text/html")
        fmt.Fprintf(w, `
<!DOCTYPE html>
<html>
<head>
    <title>ARCA-b Chat AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 10px;
            background-color: #f0f2f5;
            transition: background-color 0.3s, color 0.3s;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        body.dark {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        h1 {
            font-size: 1.5em;
            margin-bottom: 10px;
            text-align: center;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        #chat {
            border: 1px solid #ccc;
            padding: 10px;
            background-color: white;
            border-radius: 10px;
            transition: background-color 0.3s;
            min-height: 200px;
        }
        body.dark #chat {
            background-color: #2a2a2a;
            border-color: #444;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
            word-wrap: break-word;
            opacity: 0;
            animation: fadeIn 0.5s forwards;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user {
            background-color: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot {
            background-color: #e9ecef;
            color: black;
            margin-right: auto;
        }
        body.dark .user {
            background-color: #1e90ff;
        }
        body.dark .bot {
            background-color: #444;
            color: #e0e0e0;
        }
        .style-label {
            font-style: italic;
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        body.dark .style-label {
            color: #aaa;
        }
        .input-and-style-container {
            position: sticky;
            bottom: 10px;
            background-color: #f0f2f5;
            padding: 10px 0;
            z-index: 10;
            transition: background-color 0.3s;
        }
        body.dark .input-and-style-container {
            background-color: #1a1a1a;
        }
        .style-container {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        select {
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 1em;
            background-color: white;
            transition: background-color 0.3s, border-color 0.3s;
        }
        body.dark select {
            background-color: #333;
            border-color: #555;
            color: #e0e0e0;
        }
        .input-container {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }
        #input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 1em;
            transition: background-color 0.3s, border-color 0.3s;
            min-width: 200px;
        }
        body.dark #input {
            background-color: #333;
            border-color: #555;
            color: #e0e0e0;
        }
        button {
            padding: 10px 15px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }
        button:hover {
            background-color: #0056b3;
        }
        body.dark button {
            background-color: #1e90ff;
        }
        body.dark button:hover {
            background-color: #0066cc;
        }
        .typing {
            font-style: italic;
            color: #888;
            background-color: #f0f0f0;
            animation: pulse 1s infinite;
        }
        body.dark .typing {
            background-color: #555;
            color: #ccc;
        }
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        .details {
            display: none;
            margin-top: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        body.dark .details {
            background-color: #333;
            border-color: #555;
        }
        .toggle-details {
            cursor: pointer;
            color: #007bff;
            text-decoration: underline;
            margin-top: 5px;
            display: inline-block;
        }
        body.dark .toggle-details {
            color: #1e90ff;
        }
        @media (max-width: 600px) {
            h1 {
                font-size: 1.2em;
            }
            #chat-container {
                margin-bottom: 10px;
            }
            #chat {
                margin-top: 10px;
            }
            .input-and-style-container {
                padding: 5px 0;
            }
            .style-container {
                flex-direction: column;
                gap: 5px;
            }
            select {
                width: 100%;
            }
            .input-container {
                flex-direction: column;
                gap: 5px;
            }
            #input {
                width: 100%;
                font-size: 0.9em;
            }
            button {
                width: 100%;
                padding: 12px;
                font-size: 0.9em;
            }
            .button-container {
                flex-direction: column;
                gap: 5px;
            }
            .button-container button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <h1>ARCA-b Chat AI</h1>
    <p style="text-align: center; font-size: 0.9em; color: #666;">
        Nota: Le conversazioni vengono salvate temporaneamente per migliorare l'esperienza. Non inserire informazioni personali o sensibili.
    </p>
    <div class="button-container">
        <button onclick="toggleTheme()">Tema Scuro/Chiaro</button>
    </div>
    <div id="chat-container">
        <div id="chat"></div>
    </div>
    <div class="input-and-style-container">
        <div class="style-container">
            <select id="style-select">
                <option value="grok">Stile risposta informale</option>
                <option value="arca-b">Stile risposta ARCA-b</option>
            </select>
        </div>
        <div class="input-container">
            <input id="input" type="text" placeholder="Scrivi la tua domanda...">
            <button onclick="sendMessage()">Invia</button>
            <button onclick="clearChat()">Cancella Chat</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById("chat");
        const input = document.getElementById("input");
        const styleSelect = document.getElementById("style-select");

        if (localStorage.getItem("theme") === "dark") {
            document.body.classList.add("dark");
        }
        if (localStorage.getItem("style") === "arca-b") {
            styleSelect.value = "arca-b";
        } else {
            styleSelect.value = "grok";
        }

        const emojis = ["😊", "🚀", "🌟", "🎉", "🤓", "💡", "👍"];
        function getRandomEmoji() {
            return emojis[Math.floor(Math.random() * emojis.length)];
        }

        function toggleTheme() {
            document.body.classList.toggle("dark");
            localStorage.setItem("theme", document.body.classList.contains("dark") ? "dark" : "light");
        }

        function addMessage(text, isUser, style, rawResponses) {
            const div = document.createElement("div");
            if (!isUser && style) {
                const label = document.createElement("div");
                label.className = "style-label";
                label.textContent = "Risposta in stile " + (style === "grok" ? "informale" : "ARCA-b");
                chat.appendChild(label);
            }
            const messageText = (isUser ? "Tu: " : "ARCA-b: ") + text + (isUser ? "" : " " + getRandomEmoji());
            div.innerHTML = messageText.replace(/\n/g, "<br>");
            div.className = "message " + (isUser ? "user" : "bot");
            chat.appendChild(div);

            if (!isUser && rawResponses && rawResponses.trim() !== "") {
                console.log("Raw responses received:", rawResponses);
                const toggle = document.createElement("span");
                toggle.className = "toggle-details";
                toggle.textContent = "Mostra risposte originali";
                toggle.onclick = function() {
                    const details = this.nextSibling;
                    if (details.style.display === "none" || details.style.display === "") {
                        details.style.display = "block";
                        this.textContent = "Nascondi risposte originali";
                    } else {
                        details.style.display = "none";
                        this.textContent = "Mostra risposte originali";
                    }
                };
                chat.appendChild(toggle);

                const details = document.createElement("div");
                details.className = "details";
                details.innerHTML = rawResponses.replace(/\n/g, "<br>");
                chat.appendChild(details);
            } else if (!isUser) {
                console.log("No raw responses provided or empty for this message.");
            }

            chat.scrollTop = chat.scrollHeight;
        }

        function showTypingIndicator() {
            const div = document.createElement("div");
            div.id = "typing-indicator";
            div.className = "message bot typing";
            div.textContent = "ARCA-b sta scrivendo...";
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        function removeTypingIndicator() {
            const typingIndicator = document.getElementById("typing-indicator");
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }

        async function sendMessage() {
            const question = input.value.trim();
            if (!question) return;
            addMessage(question, true);
            input.value = "";

            const style = styleSelect.value;
            localStorage.setItem("style", style);

            const typingIndicator = showTypingIndicator();
            try {
                const minDisplayTime = new Promise(resolve => setTimeout(resolve, 1000));
                const response = await fetch("/ask?question=" + encodeURIComponent(question) + "&style=" + style, {
                    credentials: "include"
                });
                const answer = await Promise.all([response.json(), minDisplayTime]);
                console.log("Response received from /ask:", answer[0]);
                removeTypingIndicator();
                const rawResponses = answer[0].rawResponses || "";
                const synthesizedAnswer = answer[0].synthesized || "Errore: nessuna risposta sintetizzata.";
                addMessage(synthesizedAnswer, false, style, rawResponses);
            } catch (error) {
                console.error("Errore durante la richiesta:", error);
                removeTypingIndicator();
                addMessage("Errore: non sono riuscito a ottenere una risposta.", false, style);
            }
        }

        function clearChat() {
            fetch("/clear", {
                method: "POST",
                credentials: "include"
            }).then(() => {
                chat.innerHTML = "";
            });
        }

        input.addEventListener("keypress", function(e) {
            if (e.key === "Enter") sendMessage();
        });
    </script>
</body>
</html>
`)
    })

    // Endpoint per cancellare la chat
    http.HandleFunc("/clear", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost {
            http.Error(w, "Metodo non consentito", http.StatusMethodNotAllowed)
            return
        }
        sessionID, err := r.Cookie("session_id")
        if err != nil {
            http.Error(w, "Errore: sessione non trovata", http.StatusBadRequest)
            return
        }
        mutex.Lock()
        delete(sessions, sessionID.Value)
        mutex.Unlock()
        w.WriteHeader(http.StatusOK)
    })

    // Endpoint /ask con sintesi tramite Grok
    http.HandleFunc("/ask", func(w http.ResponseWriter, r *http.Request) {
        fmt.Println("Ricevuta richiesta su /ask")
        sessionID, err := r.Cookie("session_id")
        if err != nil {
            http.Error(w, "Errore: sessione non trovata", http.StatusBadRequest)
            return
        }

        question := r.URL.Query().Get("question")
        if question == "" {
            http.Error(w, "Errore: specifica una domanda con ?question=", http.StatusBadRequest)
            return
        }
        question = strings.TrimSpace(question)
        question = strings.ReplaceAll(question, "<", "<")
        question = strings.ReplaceAll(question, ">", ">")

        style := r.URL.Query().Get("style")
        if style == "" {
            style = "grok"
        }
        if style == "inama" {
            style = "arca-b"
        }

        mutex.Lock()
        session, exists := sessions[sessionID.Value]
        if !exists {
            session = &Session{History: []openai.ChatCompletionMessage{}}
            sessions[sessionID.Value] = session
        }
        mutex.Unlock()

        session.History = append(session.History, openai.ChatCompletionMessage{
            Role:    openai.ChatMessageRoleUser,
            Content: question,
        })

        // 1. OpenAI
        var openAIAnswer string
        if openAIKey == "" {
            openAIAnswer = "Errore: OPENAI_API_KEY non è impostata."
        } else {
            ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
            defer cancel()
            openAIResp, err := openAIClient.CreateChatCompletion(
                ctx,
                openai.ChatCompletionRequest{
                    Model:    openai.GPT3Dot5Turbo,
                    Messages: session.History,
                },
            )
            if err != nil {
                fmt.Printf("Errore con OpenAI: %v\n", err)
                openAIAnswer = "Errore: OpenAI non ha risposto."
            } else {
                openAIAnswer = openAIResp.Choices[0].Message.Content
            }
        }

        // 2. DeepSeek
        var deepSeekAnswer string
        deepSeekAnswer, err = getDeepSeekResponse(session.History)
        if err != nil {
            fmt.Printf("Errore con DeepSeek: %v\n", err)
            deepSeekAnswer = "Errore: DeepSeek non ha risposto."
        }

        // 3. Gemini
        var geminiAnswer string
        if geminiKey == "" {
            geminiAnswer = "Errore: GEMINI_API_KEY non è impostata."
        } else {
            historyForGemini := ""
            for _, msg := range session.History {
                historyForGemini += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
            }
            geminiReq, err := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key="+geminiKey,
                strings.NewReader(fmt.Sprintf(`{"contents":[{"parts":[{"text":"%s"}]}]}`, historyForGemini)))
            if err != nil {
                fmt.Printf("Errore nella creazione della richiesta a Gemini: %v\n", err)
                geminiAnswer = "Errore: Gemini non ha risposto."
            } else {
                geminiReq.Header.Set("Content-Type", "application/json")
                geminiResp, err := client.Do(geminiReq)
                if err != nil {
                    fmt.Printf("Errore con Gemini: %v\n", err)
                    geminiAnswer = "Errore: Gemini non ha risposto."
                } else {
                    defer geminiResp.Body.Close()
                    var geminiResult struct {
                        Candidates []struct {
                            Content struct {
                                Parts []struct {
                                    Text string `json:"text"`
                                } `json:"parts"`
                            } `json:"content"`
                        } `json:"candidates"`
                    }
                    if err := json.NewDecoder(geminiResp.Body).Decode(&geminiResult); err != nil {
                        fmt.Printf("Errore nel parsing di Gemini: %v\n", err)
                        geminiAnswer = "Errore: Gemini non ha risposto correttamente."
                    } else if len(geminiResult.Candidates) == 0 || len(geminiResult.Candidates[0].Content.Parts) == 0 {
                        geminiAnswer = "Errore: Nessuna risposta valida da Gemini."
                    } else {
                        geminiAnswer = geminiResult.Candidates[0].Content.Parts[0].Text
                    }
                }
            }
        }

        // 4. Costruisci rawResponses
        rawResponses := strings.Builder{}
        rawResponses.WriteString("### Risposte originali\n\n")
        rawResponses.WriteString("#### OpenAI\n")
        rawResponses.WriteString(openAIAnswer + "\n\n")
        rawResponses.WriteString("#### DeepSeek\n")
        rawResponses.WriteString(deepSeekAnswer + "\n\n")
        rawResponses.WriteString("#### Gemini\n")
        rawResponses.WriteString(geminiAnswer + "\n\n")

        // 5. Sintesi tramite Grok con fallback
        var synthesizedAnswer string
        // Escludi risposte con errori dal prompt di sintesi
        synthesisParts := []string{}
        if !strings.Contains(openAIAnswer, "Errore") {
            synthesisParts = append(synthesisParts, fmt.Sprintf("OpenAI: %s", openAIAnswer))
        }
        if !strings.Contains(deepSeekAnswer, "Errore") {
            synthesisParts = append(synthesisParts, fmt.Sprintf("DeepSeek: %s", deepSeekAnswer))
        }
        if !strings.Contains(geminiAnswer, "Errore") {
            synthesisParts = append(synthesisParts, fmt.Sprintf("Gemini: %s", geminiAnswer))
        }

        if len(synthesisParts) == 0 {
            synthesizedAnswer = "Errore: nessuna risposta valida da sintetizzare."
        } else {
            synthesisPrompt := fmt.Sprintf(
                "The user asked: '%s'. Synthesize the following answers into a single, comprehensive response that directly addresses the user's question. Integrate scientific, cultural, historical, and technological perspectives to reflect a global digital knowledge. Avoid bias, censorship, and propaganda. If the answers are off-topic or incomplete, provide a correct and complete response based on the question. Provide a clear and concise answer in %s style:\n\n%s",
                question, style, strings.Join(synthesisParts, "\n\n"),
            )
            synthesizedAnswer, err = getGrokResponse(grokKey, client, synthesisPrompt)
            if err != nil {
                fmt.Printf("Errore nella sintesi con Grok: %v\n", err)
                // Fallback: usa la prima risposta valida disponibile
                if !strings.Contains(openAIAnswer, "Errore") {
                    synthesizedAnswer = openAIAnswer + " (Nota: sintesi non disponibile, risposta di OpenAI.)"
                } else if !strings.Contains(geminiAnswer, "Errore") {
                    synthesizedAnswer = geminiAnswer + " (Nota: sintesi non disponibile, risposta di Gemini.)"
                } else if !strings.Contains(deepSeekAnswer, "Errore") {
                    synthesizedAnswer = deepSeekAnswer + " (Nota: sintesi non disponibile, risposta di DeepSeek.)"
                } else {
                    synthesizedAnswer = "Errore: non sono riuscito a sintetizzare le risposte."
                }
            }
        }

        // 6. Risposta finale
        response := map[string]string{
            "synthesized":  synthesizedAnswer,
            "rawResponses": rawResponses.String(),
        }
        fmt.Println("Invio risposta JSON:", response)

        mutex.Lock()
        session.History = append(session.History, openai.ChatCompletionMessage{
            Role:    openai.ChatMessageRoleAssistant,
            Content: synthesizedAnswer,
        })
        mutex.Unlock()

        w.Header().Set("Content-Type", "application/json")
        if err := json.NewEncoder(w).Encode(response); err != nil {
            fmt.Printf("Errore nell'invio della risposta JSON: %v\n", err)
            http.Error(w, "Errore interno del server", http.StatusInternalServerError)
            return
        }
    })

    fmt.Printf("Server in ascolto sulla porta %s...\n", port)
    if err := http.ListenAndServe(":"+port, nil); err != nil {
        fmt.Printf("Errore nell'avvio del server: %v\n", err)
        os.Exit(1)
    }
}

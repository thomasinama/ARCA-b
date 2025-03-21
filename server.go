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

func main() {
    // Carica le chiavi API dalle variabili d'ambiente
    openAIKey := os.Getenv("OPENAI_API_KEY")
    deepSeekKey := os.Getenv("DEEPSEEK_API_KEY")
    geminiKey := os.Getenv("GEMINI_API_KEY")

    // Log per verificare le chiavi API
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

    // Carica la porta da usare
    port := os.Getenv("PORT")
    if port == "" {
        fmt.Println("PORT non specificata, uso default :10000")
        port = "10000" // Default per test locali
    }

    openAIClient := openai.NewClient(openAIKey)

    // Crea un client HTTP con timeout
    client := &http.Client{
        Timeout: 20 * time.Second, // Aumentato a 20 secondi
    }

    getDeepSeekResponse := func(messages []openai.ChatCompletionMessage) (string, error) {
        if deepSeekKey == "" {
            return "", fmt.Errorf("DEEPSEEK_API_KEY non è impostata")
        }
        // Converti lo storico in formato DeepSeek
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

        resp, err := client.Do(req) // Usa il client con timeout
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
        fmt.Println("Ricevuta richiesta su /")
        // Imposta un cookie per identificare la sessione
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
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-YOUR-MEASUREMENT-ID"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-YOUR-MEASUREMENT-ID');
    </script>
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
                <option value="inama">Stile risposta Inama</option>
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

        // Carica il tema salvato
        if (localStorage.getItem("theme") === "dark") {
            document.body.classList.add("dark");
        }

        // Carica lo stile salvato
        if (localStorage.getItem("style") === "inama") {
            styleSelect.value = "inama";
        } else {
            styleSelect.value = "grok";
        }

        function toggleTheme() {
            console.log("Toggling theme...");
            document.body.classList.toggle("dark");
            localStorage.setItem("theme", document.body.classList.contains("dark") ? "dark" : "light");
            console.log("Theme set to: " + (document.body.classList.contains("dark") ? "dark" : "light"));
        }

        function addMessage(text, isUser, style) {
            const div = document.createElement("div");
            if (!isUser && style) {
                const label = document.createElement("div");
                label.className = "style-label";
                label.textContent = "Risposta in stile " + (style === "grok" ? "informale" : "Inama");
                chat.appendChild(label);
            }
            div.textContent = (isUser ? "Tu: " : "ARCA-b: ") + text;
            div.className = "message " + (isUser ? "user" : "bot");
            chat.appendChild(div);
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

            // Mostra l'indicatore "Sta scrivendo..."
            const typingIndicator = showTypingIndicator();

            try {
                // Assicura che l'indicatore sia visibile per almeno 1 secondo
                const minDisplayTime = new Promise(resolve => setTimeout(resolve, 1000));
                const response = await fetch("/ask?question=" + encodeURIComponent(question) + "&style=" + style, {
                    credentials: "include"
                });
                const answer = await Promise.all([response.json(), minDisplayTime]);
                // Rimuove l'indicatore "Sta scrivendo..." e mostra la risposta
                removeTypingIndicator();
                addMessage(answer[0], false, style);
            } catch (error) {
                // In caso di errore, rimuove l'indicatore e mostra un messaggio di errore
                removeTypingIndicator();
                addMessage("Errore: non sono riuscito a ottenere una risposta. Riprova più tardi.", false, style);
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

    // Endpoint /ask
    http.HandleFunc("/ask", func(w http.ResponseWriter, r *http.Request) {
        fmt.Println("Ricevuta richiesta su /ask")
        // Recupera il sessionID dal cookie
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
        // Sanitizza l'input
        question = strings.TrimSpace(question)
        question = strings.ReplaceAll(question, "<", "<")
        question = strings.ReplaceAll(question, ">", ">")

        style := r.URL.Query().Get("style")
        if style == "" {
            style = "grok" // Default a Grok style
        }

        // Recupera o crea la sessione
        mutex.Lock()
        session, exists := sessions[sessionID.Value]
        if !exists {
            session = &Session{History: []openai.ChatCompletionMessage{}}
            sessions[sessionID.Value] = session
        }
        mutex.Unlock()

        // Aggiungi la domanda allo storico
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
                openAIAnswer = "Errore: OpenAI non ha risposto (timeout o errore di rete). Prova a riformulare la domanda o riprova più tardi."
            } else {
                openAIAnswer = openAIResp.Choices[0].Message.Content
            }
        }

        // 2. DeepSeek
        var deepSeekAnswer string
        deepSeekAnswer, err = getDeepSeekResponse(session.History)
        if err != nil {
            fmt.Printf("Errore con DeepSeek: %v\n", err)
            deepSeekAnswer = "Errore: DeepSeek non ha risposto (timeout o errore di rete). Prova a riformulare la domanda o riprova più tardi."
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
                geminiAnswer = "Errore: Gemini non ha risposto (errore nella richiesta). Prova a riformulare la domanda o riprova più tardi."
            } else {
                geminiReq.Header.Set("Content-Type", "application/json")
                geminiResp, err := client.Do(geminiReq)
                if err != nil {
                    fmt.Printf("Errore con Gemini: %v\n", err)
                    geminiAnswer = "Errore: Gemini non ha risposto (timeout o errore di rete). Prova a riformulare la domanda o riprova più tardi."
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
                        geminiAnswer = "Errore: Gemini non ha risposto correttamente. Prova a riformulare la domanda o riprova più tardi."
                    } else if len(geminiResult.Candidates) == 0 || len(geminiResult.Candidates[0].Content.Parts) == 0 {
                        geminiAnswer = "Errore: Nessuna risposta valida da Gemini."
                    } else {
                        geminiAnswer = geminiResult.Candidates[0].Content.Parts[0].Text
                    }
                }
            }
        }

        // 4. Rielabora con OpenAI, includendo lo storico
        historyPrompt := "Storico della conversazione:\n"
        for _, msg := range session.History {
            historyPrompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
        }
        var prompt string
        if style == "grok" {
            prompt = fmt.Sprintf("%s\nNuova domanda: '%s'\nTutte e tre le AI (OpenAI, DeepSeek, Gemini) hanno contribuito. Usa queste risposte senza mostrarle direttamente: OpenAI: %s, DeepSeek: %s, Gemini: %s. Fornisci una risposta esaustiva, dettagliata e utile che integri i loro contributi con molti dettagli, mantenendo un tono chiaro, amichevole e sobrio, senza elementi satirici. Assicurati di rispondere in modo contestuale, considerando lo storico della conversazione. Se una delle risposte contiene un errore, ignoralo e usa le altre risposte per costruire una risposta coerente.", historyPrompt, question, openAIAnswer, deepSeekAnswer, geminiAnswer)
        } else {
            prompt = fmt.Sprintf("%s\nNuova domanda: '%s'\nTutte e tre le AI (OpenAI, DeepSeek, Gemini) hanno contribuito. Usa queste risposte senza mostrarle direttamente: OpenAI: %s, DeepSeek: %s, Gemini: %s. Fornisci una risposta esaustiva, dettagliata e utile che integri i loro contributi con molti dettagli, mantenendo un tono informale, eclettico, amichevole, diretto, quasi satirico ma tagliente ed acuto, in stile Inama. Assicurati di rispondere in modo contestuale, considerando lo storico della conversazione. Se una delle risposte contiene un errore, ignoralo e usa le altre risposte per costruire una risposta coerente.", historyPrompt, question, openAIAnswer, deepSeekAnswer, geminiAnswer)
        }
        var finalAnswer string
        if openAIKey == "" {
            finalAnswer = "Errore: OPENAI_API_KEY non è impostata, non posso rielaborare le risposte."
        } else {
            ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
            defer cancel()
            finalResp, err := openAIClient.CreateChatCompletion(
                ctx,
                openai.ChatCompletionRequest{
                    Model:    openai.GPT3Dot5Turbo,
                    Messages: []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: prompt}},
                },
            )
            if err != nil {
                fmt.Printf("Errore nella rielaborazione con OpenAI: %v\n", err)
                // Fallback: usa le risposte delle altre API
                finalAnswer = "Errore: non sono riuscito a rielaborare le risposte con OpenAI. Ecco una sintesi delle risposte disponibili:\n"
                if !strings.Contains(openAIAnswer, "Errore") {
                    finalAnswer += "OpenAI: " + openAIAnswer + "\n"
                } else {
                    finalAnswer += "OpenAI: (non disponibile)\n"
                }
                if !strings.Contains(deepSeekAnswer, "Errore") {
                    finalAnswer += "DeepSeek: " + deepSeekAnswer + "\n"
                } else {
                    finalAnswer += "DeepSeek: (non disponibile)\n"
                }
                if !strings.Contains(geminiAnswer, "Errore") {
                    finalAnswer += "Gemini: " + geminiAnswer + "\n"
                } else {
                    finalAnswer += "Gemini: (non disponibile)\n"
                }
                if finalAnswer == "Errore: non sono riuscito a rielaborare le risposte con OpenAI. Ecco una sintesi delle risposte disponibili:\nOpenAI: (non disponibile)\nDeepSeek: (non disponibile)\nGemini: (non disponibile)\n" {
                    finalAnswer = "Errore: nessuna delle AI ha risposto correttamente. Prova a riformulare la domanda o riprova più tardi."
                }
            } else {
                finalAnswer = finalResp.Choices[0].Message.Content
            }
        }

        // Aggiungi la risposta allo storico
        mutex.Lock()
        session.History = append(session.History, openai.ChatCompletionMessage{
            Role:    openai.ChatMessageRoleAssistant,
            Content: finalAnswer,
        })
        mutex.Unlock()

        json.NewEncoder(w).Encode(finalAnswer)
    })

    fmt.Printf("Server in ascolto sulla porta %s...\n", port)
    if err := http.ListenAndServe(":"+port, nil); err != nil {
        fmt.Printf("Errore nell'avvio del server: %v\n", err)
        os.Exit(1)
    }
}

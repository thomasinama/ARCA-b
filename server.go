kpackage main

import (
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "strings"
    "sync"

    "github.com/google/uuid"
    "github.com/sashabaranov/go-openai"
    "context"
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

    if openAIKey == "" || deepSeekKey == "" || geminiKey == "" {
        fmt.Println("Errore: una o più chiavi API non sono impostate")
        return
    }

    openAIClient := openai.NewClient(openAIKey)

    getDeepSeekResponse := func(messages []openai.ChatCompletionMessage) (string, error) {
        // Converti lo storico in formato DeepSeek
        var deepSeekMessages []map[string]string
        for _, msg := range messages {
            deepSeekMessages = append(deepSeekMessages, map[string]string{
                "role":    msg.Role,
                "content": msg.Content,
            })
        }

        body, _ := json.Marshal(map[string]interface{}{
            "model":    "deepseek-chat",
            "messages": deepSeekMessages,
        })
        req, _ := http.NewRequest("POST", "https://api.deepseek.com/v1/chat/completions", strings.NewReader(string(body)))
        req.Header.Set("Authorization", "Bearer "+deepSeekKey)
        req.Header.Set("Content-Type", "application/json")
        resp, err := http.DefaultClient.Do(req)
        if err != nil {
            return "", err
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

    // Serve la pagina HTML
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
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
    <title>ARCA-b Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f2f5;
            transition: background-color 0.3s, color 0.3s;
        }
        body.dark {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        #chat {
            border: 1px solid #ccc;
            padding: 10px;
            height: 400px;
            overflow-y: scroll;
            background-color: white;
            border-radius: 10px;
            margin-bottom: 20px;
            transition: background-color 0.3s;
        }
        body.dark #chat {
            background-color: #2a2a2a;
            border-color: #444;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
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
        #input {
            width: 70%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-right: 10px;
            transition: background-color 0.3s, border-color 0.3s;
        }
        body.dark #input {
            background-color: #333;
            border-color: #555;
            color: #e0e0e0;
        }
        button {
            padding: 8px 15px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .input-container {
            display: flex;
            align-items: center;
        }
    </style>
</head>
<body>
    <h1>ARCA-b Chat</h1>
    <button onclick="toggleTheme()">Cambia Tema</button>
    <div id="chat"></div>
    <div class="input-container">
        <input id="input" type="text" placeholder="Scrivi la tua domanda...">
        <button onclick="sendMessage()">Invia</button>
        <button onclick="clearChat()">Cancella Chat</button>
    </div>
    <script>
        const chat = document.getElementById("chat");
        const input = document.getElementById("input");

        // Carica il tema salvato
        if (localStorage.getItem("theme") === "dark") {
            document.body.classList.add("dark");
        }

        function toggleTheme() {
            document.body.classList.toggle("dark");
            localStorage.setItem("theme", document.body.classList.contains("dark") ? "dark" : "light");
        }

        function addMessage(text, isUser) {
            const div = document.createElement("div");
            div.textContent = text;
            div.className = "message " + (isUser ? "user" : "bot");
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        async function sendMessage() {
            const question = input.value.trim();
            if (!question) return;
            addMessage("Tu: " + question, true);
            input.value = "";

            const response = await fetch("/ask?question=" + encodeURIComponent(question), {
                credentials: "include"
            });
            const answer = await response.json();
            addMessage("ARCA-b: " + answer, false);
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
        openAIResp, err := openAIClient.CreateChatCompletion(
            context.Background(),
            openai.ChatCompletionRequest{
                Model:    openai.GPT3Dot5Turbo,
                Messages: session.History,
            },
        )
        if err != nil {
            openAIAnswer = "Errore: OpenAI non ha risposto. Prova a riformulare la domanda o riprova più tardi."
        } else {
            openAIAnswer = openAIResp.Choices[0].Message.Content
        }

        // 2. DeepSeek
        var deepSeekAnswer string
        deepSeekAnswer, err = getDeepSeekResponse(session.History)
        if err != nil {
            deepSeekAnswer = "Errore: DeepSeek non ha risposto. Prova a riformulare la domanda o riprova più tardi."
        }

        // 3. Gemini
        var geminiAnswer string
        historyForGemini := ""
        for _, msg := range session.History {
            historyForGemini += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
        }
        geminiReq, _ := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key="+geminiKey,
            strings.NewReader(fmt.Sprintf(`{"contents":[{"parts":[{"text":"%s"}]}]}`, historyForGemini)))
        geminiReq.Header.Set("Content-Type", "application/json")
        geminiResp, err := http.DefaultClient.Do(geminiReq)
        if err != nil {
            geminiAnswer = "Errore: Gemini non ha risposto. Prova a riformulare la domanda o riprova più tardi."
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
                geminiAnswer = "Errore: Gemini non ha risposto correttamente. Prova a riformulare la domanda o riprova più tardi."
            } else if len(geminiResult.Candidates) == 0 || len(geminiResult.Candidates[0].Content.Parts) == 0 {
                geminiAnswer = "Errore: Nessuna risposta valida da Gemini."
            } else {
                geminiAnswer = geminiResult.Candidates[0].Content.Parts[0].Text
            }
        }

        // 4. Rielabora con OpenAI, includendo lo storico
        historyPrompt := "Storico della conversazione:\n"
        for _, msg := range session.History {
            historyPrompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
        }
        prompt := fmt.Sprintf("%s\nNuova domanda: '%s'\nTutte e tre le AI (OpenAI, DeepSeek, Gemini) hanno contribuito. Usa queste risposte senza mostrarle direttamente: OpenAI: %s, DeepSeek: %s, Gemini: %s. Fornisci una risposta esaustiva, dettagliata e utile che integri i loro contributi con molti dettagli, mantenendo un tono chiaro e amichevole. Assicurati di rispondere in modo contestuale, considerando lo storico della conversazione.", historyPrompt, question, openAIAnswer, deepSeekAnswer, geminiAnswer)
        finalResp, err := openAIClient.CreateChatCompletion(
            context.Background(),
            openai.ChatCompletionRequest{
                Model:    openai.GPT3Dot5Turbo,
                Messages: []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: prompt}},
            },
        )
        if err != nil {
            http.Error(w, "Errore nella rielaborazione con OpenAI: "+err.Error(), http.StatusInternalServerError)
            return
        }
        finalAnswer := finalResp.Choices[0].Message.Content

        // Aggiungi la risposta allo storico
        mutex.Lock()
        session.History = append(session.History, openai.ChatCompletionMessage{
            Role:    openai.ChatMessageRoleAssistant,
            Content: finalAnswer,
        })
        mutex.Unlock()

        json.NewEncoder(w).Encode(finalAnswer)
    })

    fmt.Println("Server in ascolto su :10000...")
    http.ListenAndServe(":10000", nil)
}

package main

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

    getDeepSeekResponse := func(message string) (string, error) {
        req, _ := http.NewRequest("POST", "https://api.deepseek.com/v1/chat/completions", strings.NewReader(
            fmt.Sprintf(`{"model":"deepseek-chat","messages":[{"role":"user","content":"%s"}]}`, message),
        ))
        req.Header.Set("Authorization", "Bearer "+deepSeekKey)
        req.Header.Set("Content-Type", "application/json")
        resp, err := http.DefaultClient.Do(req)
        if err != nil {
            return "", err
        }
        defer resp.Body.Close()

        body, err := io.ReadAll(resp.Body)
        if err != nil {
            return "", fmt.Errorf("errore nella lettura della risposta: %v", err)
        }
        if resp.StatusCode != http.StatusOK {
            return "", fmt.Errorf("risposta non valida da DeepSeek (status %d): %s", resp.StatusCode, string(body))
        }

        var result struct {
            Choices []struct {
                Message struct {
                    Content string `json:"content"`
                } `json:"message"`
            } `json:"choices"`
        }
        if err := json.Unmarshal(body, &result); err != nil {
            return "", fmt.Errorf("errore nel parsing JSON: %v, risposta grezza: %s", err, string(body))
        }
        if len(result.Choices) > 0 {
            return result.Choices[0].Message.Content, nil
        }
        return "", fmt.Errorf("nessuna risposta valida da DeepSeek: %s", string(body))
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
        body { font-family: Arial, sans-serif; margin: 20px; }
        #chat { border: 1px solid #ccc; padding: 10px; height: 400px; overflow-y: scroll; }
        #input { width: 80%; padding: 5px; }
        button { padding: 5px 10px; }
    </style>
</head>
<body>
    <h1>ARCA-b Chat</h1>
    <div id="chat"></div>
    <input id="input" type="text" placeholder="Scrivi la tua domanda...">
    <button onclick="sendMessage()">Invia</button>
    <script>
        const chat = document.getElementById("chat");
        const input = document.getElementById("input");

        function addMessage(text, isUser) {
            const div = document.createElement("div");
            div.textContent = text;
            div.style.color = isUser ? "blue" : "black";
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

        input.addEventListener("keypress", function(e) {
            if (e.key === "Enter") sendMessage();
        });
    </script>
</body>
</html>
`)
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
        openAIResp, err := openAIClient.CreateChatCompletion(
            context.Background(),
            openai.ChatCompletionRequest{
                Model:    openai.GPT3Dot5Turbo,
                Messages: []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: question}},
            },
        )
        if err != nil {
            http.Error(w, "Errore con OpenAI: "+err.Error(), http.StatusInternalServerError)
            return
        }
        openAIAnswer := openAIResp.Choices[0].Message.Content

        // 2. DeepSeek
        deepSeekAnswer, err := getDeepSeekResponse(question)
        if err != nil {
            http.Error(w, "Errore con DeepSeek: "+err.Error(), http.StatusInternalServerError)
            return
        }

        // 3. Gemini
        geminiReq, _ := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key="+geminiKey,
            strings.NewReader(fmt.Sprintf(`{"contents":[{"parts":[{"text":"%s"}]}]}`, question)))
        geminiReq.Header.Set("Content-Type", "application/json")
        geminiResp, err := http.DefaultClient.Do(geminiReq)
        if err != nil {
            http.Error(w, "Errore con Gemini: "+err.Error(), http.StatusInternalServerError)
            return
        }
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
            http.Error(w, "Errore nel parsing di Gemini: "+err.Error(), http.StatusInternalServerError)
            return
        }
        if len(geminiResult.Candidates) == 0 || len(geminiResult.Candidates[0].Content.Parts) == 0 {
            http.Error(w, "Nessuna risposta da Gemini", http.StatusInternalServerError)
            return
        }
        geminiAnswer := geminiResult.Candidates[0].Content.Parts[0].Text

        // 4. Rielabora con OpenAI, includendo lo storico
        historyPrompt := "Storico della conversazione:\n"
        for _, msg := range session.History {
            historyPrompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
        }
        prompt := fmt.Sprintf("%s\nNuova domanda: '%s'\nTutte e tre le AI (OpenAI, DeepSeek, Gemini) hanno contribuito. Usa queste risposte senza mostrarle direttamente: OpenAI: %s, DeepSeek: %s, Gemini: %s. Fornisci una risposta esaustiva, dettagliata e utile che integri i loro contributi con molti dettagli, mantenendo un tono chiaro e amichevole.", historyPrompt, question, openAIAnswer, deepSeekAnswer, geminiAnswer)
        finalResp, err := openAIClient.CreateChatCompletion(
            context.Background(),
            openai.ChatCompletionRequest{
                Model: openai.GPT3Dot5Turbo,
                Messages: []openai.ChatCompletionMessage{
                    {Role: openai.ChatMessageRoleUser, Content: prompt},
                },
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

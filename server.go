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

// getDeepInfraResponse per sintetizzare le risposte usando l'API di DeepInfra
func getDeepInfraResponse(deepInfraKey string, client *http.Client, prompt string) (string, error) {
    if deepInfraKey == "" {
        return "", fmt.Errorf("DEEPINFRA_API_KEY non è impostata")
    }

    // Sanifica il prompt
    prompt = strings.ReplaceAll(prompt, "\n", " ")
    prompt = strings.ReplaceAll(prompt, "\"", "\\\"")
    logLimit := 100
    if len(prompt) < logLimit {
        logLimit = len(prompt)
    }
    fmt.Println("Invio richiesta a DeepInfra con prompt (prime 100 chars):", prompt[:logLimit], "...")

    // Prepara la richiesta per DeepInfra (usiamo il modello Meta-Llama-3-8B-Instruct)
    payload := fmt.Sprintf(`{"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "user", "content": "%s"}], "max_tokens": 1000, "temperature": 0.7}`, prompt)
    req, err := http.NewRequest("POST", "https://api.deepinfra.com/v1/openai/chat/completions", strings.NewReader(payload))
    if err != nil {
        return "", fmt.Errorf("errore nella creazione della richiesta a DeepInfra: %v", err)
    }

    // Imposta gli header
    req.Header.Set("Authorization", "Bearer "+deepInfraKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")

    // Esegui la richiesta con retry
    var resp *http.Response
    for attempt := 1; attempt <= 3; attempt++ {
        resp, err = client.Do(req)
        if err == nil {
            break
        }
        fmt.Printf("Errore con DeepInfra (tentativo %d): %v\n", attempt, err)
        time.Sleep(time.Second * time.Duration(attempt))
    }
    if err != nil {
        return "", fmt.Errorf("errore con DeepInfra dopo 3 tentativi: %v", err)
    }
    defer resp.Body.Close()

    // Leggi la risposta
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", fmt.Errorf("errore nella lettura della risposta di DeepInfra: %v", err)
    }
    fmt.Println("Risposta grezza da DeepInfra (status %d): %s", resp.StatusCode, string(body))

    // Parsa la risposta JSON
    var deepInfraResult struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
        Error string `json:"error"`
    }
    if err := json.Unmarshal(body, &deepInfraResult); err != nil {
        return "", fmt.Errorf("errore nel parsing di DeepInfra: %v, risposta grezza: %s", err, string(body))
    }
    if deepInfraResult.Error != "" {
        return "", fmt.Errorf("errore da DeepInfra: %s", deepInfraResult.Error)
    }
    if len(deepInfraResult.Choices) == 0 || deepInfraResult.Choices[0].Message.Content == "" {
        return "", fmt.Errorf("nessuna risposta valida da DeepInfra: %s", string(body))
    }

    return deepInfraResult.Choices[0].Message.Content, nil
}

// getAIMLAPIResponse per sintetizzare le risposte usando l'API di AIMLAPI
func getAIMLAPIResponse(aimlKey string, client *http.Client, prompt string) (string, error) {
    if aimlKey == "" {
        return "", fmt.Errorf("AIMLAPI_API_KEY non è impostata")
    }

    // Sanifica il prompt
    prompt = strings.ReplaceAll(prompt, "\n", " ")
    prompt = strings.ReplaceAll(prompt, "\"", "\\\"")
    logLimit := 100
    if len(prompt) < logLimit {
        logLimit = len(prompt)
    }
    fmt.Println("Invio richiesta a AIMLAPI con prompt (prime 100 chars):", prompt[:logLimit], "...")

    // Prepara la richiesta per AIMLAPI
    payload := fmt.Sprintf(`{"model": "Grok", "messages": [{"role": "user", "content": "%s"}]}`, prompt)
    req, err := http.NewRequest("POST", "https://api.aimlapi.com/v1/chat/completions", strings.NewReader(payload))
    if err != nil {
        return "", fmt.Errorf("errore nella creazione della richiesta a AIMLAPI: %v", err)
    }

    // Imposta gli header
    req.Header.Set("Authorization", "Bearer "+aimlKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")

    // Esegui la richiesta con retry
    var resp *http.Response
    for attempt := 1; attempt <= 3; attempt++ {
        resp, err = client.Do(req)
        if err == nil {
            break
        }
        fmt.Printf("Errore con AIMLAPI (tentativo %d): %v\n", attempt, err)
        time.Sleep(time.Second * time.Duration(attempt))
    }
    if err != nil {
        return "", fmt.Errorf("errore con AIMLAPI dopo 3 tentativi: %v", err)
    }
    defer resp.Body.Close()

    // Leggi la risposta
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", fmt.Errorf("errore nella lettura della risposta di AIMLAPI: %v", err)
    }
    fmt.Println("Risposta grezza da AIMLAPI (status %d): %s", resp.StatusCode, string(body))

    // Parsa la risposta JSON
    var aimlResult struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
        Error string `json:"error"`
    }
    if err := json.Unmarshal(body, &aimlResult); err != nil {
        return "", fmt.Errorf("errore nel parsing di AIMLAPI: %v, risposta grezza: %s", err, string(body))
    }
    if aimlResult.Error != "" {
        return "", fmt.Errorf("errore da AIMLAPI: %s", aimlResult.Error)
    }
    if len(aimlResult.Choices) == 0 || aimlResult.Choices[0].Message.Content == "" {
        return "", fmt.Errorf("nessuna risposta valida da AIMLAPI: %s", string(body))
    }

    return aimlResult.Choices[0].Message.Content, nil
}

// getHuggingFaceResponse per sintetizzare le risposte usando l'API di Hugging Face
func getHuggingFaceResponse(hfKey string, client *http.Client, prompt string) (string, error) {
    if hfKey == "" {
        return "", fmt.Errorf("HUGGINGFACE_API_KEY non è impostata")
    }

    // Sanifica il prompt
    prompt = strings.ReplaceAll(prompt, "\n", " ")
    prompt = strings.ReplaceAll(prompt, "\"", "\\\"")
    logLimit := 100
    if len(prompt) < logLimit {
        logLimit = len(prompt)
    }
    fmt.Println("Invio richiesta a Hugging Face con prompt (prime 100 chars):", prompt[:logLimit], "...")

    // Prepara la richiesta per Hugging Face (usiamo DistilGPT-2)
    payload := fmt.Sprintf(`{"inputs": "%s", "parameters": {"max_length": 500, "temperature": 0.7, "top_p": 0.9}}`, prompt)
    req, err := http.NewRequest("POST", "https://api-inference.huggingface.co/models/distilgpt2", strings.NewReader(payload))
    if err != nil {
        return "", fmt.Errorf("errore nella creazione della richiesta a Hugging Face: %v", err)
    }

    // Imposta gli header
    req.Header.Set("Authorization", "Bearer "+hfKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")

    // Esegui la richiesta con retry
    var resp *http.Response
    for attempt := 1; attempt <= 3; attempt++ {
        resp, err = client.Do(req)
        if err == nil {
            break
        }
        fmt.Printf("Errore con Hugging Face (tentativo %d): %v\n", attempt, err)
        time.Sleep(time.Second * time.Duration(attempt))
    }
    if err != nil {
        return "", fmt.Errorf("errore con Hugging Face dopo 3 tentativi: %v", err)
    }
    defer resp.Body.Close()

    // Leggi la risposta
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", fmt.Errorf("errore nella lettura della risposta di Hugging Face: %v", err)
    }
    fmt.Println("Risposta grezza da Hugging Face (status %d): %s", resp.StatusCode, string(body))

    // Parsa la risposta JSON
    var hfResult []struct {
        GeneratedText string `json:"generated_text"`
    }
    if err := json.Unmarshal(body, &hfResult); err != nil {
        // Controlla se è un errore specifico
        var errorResult struct {
            Error string `json:"error"`
        }
        if json.Unmarshal(body, &errorResult) == nil && errorResult.Error != "" {
            return "", fmt.Errorf("errore da Hugging Face: %s", errorResult.Error)
        }
        return "", fmt.Errorf("errore nel parsing di Hugging Face: %v, risposta grezza: %s", err, string(body))
    }
    if len(hfResult) == 0 || hfResult[0].GeneratedText == "" {
        return "", fmt.Errorf("nessuna risposta valida da Hugging Face: %s", string(body))
    }

    // La risposta include il prompt, quindi rimuoviamolo
    generatedText := strings.TrimPrefix(hfResult[0].GeneratedText, prompt)
    return strings.TrimSpace(generatedText), nil
}

func main() {
    // Carica le chiavi API dalle variabili d'ambiente
    openAIKey := os.Getenv("OPENAI_API_KEY")
    deepSeekKey := os.Getenv("DEEPSEEK_API_KEY")
    geminiKey := os.Getenv("GEMINI_API_KEY")
    deepInfraKey := os.Getenv("DEEPINFRA_API_KEY")     // Chiave API per DeepInfra
    aimlKey := os.Getenv("AIMLAPI_API_KEY")           // Chiave API per AIMLAPI
    huggingFaceKey := os.Getenv("HUGGINGFACE_API_KEY") // Chiave API per Hugging Face

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
    if deepInfraKey == "" {
        fmt.Println("Errore: DEEPINFRA_API_KEY non è impostata")
    } else {
        fmt.Println("DEEPINFRA_API_KEY caricata correttamente")
    }
    if aimlKey == "" {
        fmt.Println("Errore: AIMLAPI_API_KEY non è impostata")
    } else {
        fmt.Println("AIMLAPI_API_KEY caricata correttamente")
    }
    if huggingFaceKey == "" {
        fmt.Println("Errore: HUGGINGFACE_API_KEY non è impostata")
    } else {
        fmt.Println("HUGGINGFACE_API_KEY caricata correttamente")
    }

    // Imposta la porta: usa la variabile d'ambiente PORT o default a 8080 per test locali
    port := os.Getenv("PORT")
    if port == "" {
        fmt.Println("PORT non specificata, uso default :8080")
        port = "8080" // Default per test locali (modificato da 80 a 8080 per evitare problemi di permessi)
    } else {
        fmt.Printf("PORT specificata dall'ambiente: %s\n", port)
    }

    openAIClient := openai.NewClient(openAIKey)
    client := &http.Client{
        Timeout: 30 * time.Second, // Aumentato il timeout per DeepSeek
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
        // Messaggio chiaro senza errori di formattazione
        fmt.Fprint(w, "ARCA-b Chat AI is running on arcab-global-ai.org")
    })

    // Serve la pagina HTML
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
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
        Note: Conversations are temporarily saved to improve the experience. Do not enter personal or sensitive information.
    </p>
    <div class="button-container">
        <button onclick="toggleTheme()">Dark/Light Theme</button>
        <a href="/donate"><button>Donate</button></a>
    </div>
    <div id="chat-container">
        <div id="chat"></div>
    </div>
    <div class="input-and-style-container">
        <div class="style-container">
            <select id="style-select">
                <option value="grok">Informal response style</option>
                <option value="arca-b">ARCA-b response style</option>
            </select>
        </div>
        <div class="input-container">
            <input id="input" type="text" placeholder="Write your question...">
            <button onclick="sendMessage()">Send</button>
            <button onclick="clearChat()">Clear Chat</button>
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

        function toggleTheme() {
            document.body.classList.toggle("dark");
            localStorage.setItem("theme", document.body.classList.contains("dark") ? "dark" : "light");
        }

        function addMessage(text, isUser, style, rawResponses) {
            const div = document.createElement("div");
            if (!isUser && style) {
                const label = document.createElement("div");
                label.className = "style-label";
                label.textContent = "Response in " + (style === "grok" ? "informal" : "ARCA-b") + " style";
                chat.appendChild(label);
            }
            const messageText = (isUser ? "You: " : "ARCA-b: ") + text;
            div.innerHTML = messageText.replace(/\n/g, "<br>");
            div.className = "message " + (isUser ? "user" : "bot");
            chat.appendChild(div);

            if (!isUser && rawResponses && rawResponses.trim() !== "") {
                console.log("Raw responses received:", rawResponses);
                const toggle = document.createElement("span");
                toggle.className = "toggle-details";
                toggle.textContent = "Show original responses";
                toggle.onclick = function() {
                    const details = this.nextSibling;
                    if (details.style.display === "none" || details.style.display === "") {
                        details.style.display = "block";
                        this.textContent = "Hide original responses";
                    } else {
                        details.style.display = "none";
                        this.textContent = "Show original responses";
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
            div.textContent = "ARCA-b is typing...";
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
                const synthesizedAnswer = answer[0].synthesized || "Error: No synthesized response.";
                addMessage(synthesizedAnswer, false, style, rawResponses);
            } catch (error) {
                console.error("Error during request:", error);
                removeTypingIndicator();
                addMessage("Error: I couldn't get a response.", false, style);
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

    // Endpoint per la pagina di donazioni in inglese senza emoji
    http.HandleFunc("/donate", func(w http.ResponseWriter, r *http.Request) {
        fmt.Println("Received request on /donate")
        w.Header().Set("Content-Type", "text/html")
        fmt.Fprintf(w, `<!DOCTYPE html>
<html>
<head>
    <title>Donations - ARCA-b Chat AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; text-align: center; }
        button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1em; margin: 10px; }
        button:hover { background-color: #0056b3; }
        .crypto-address { word-wrap: break-word; font-family: monospace; background-color: #f0f0f0; padding: 5px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Support ARCA-b Chat AI</h1>
    <p>Your donations help us improve the project and maintain a service free from censorship and propaganda. Thank you!</p>
    
    <h2>Fiat Donations</h2>
    <a href="https://www.paypal.com/pool/9dmdhEacmM?sr=ancr" target="_blank"><button>PayPal</button></a>
    
    <h2>Crypto Donations</h2>
    <p><strong>USDT (Ethereum):</strong><br><span class="crypto-address">0x71ECB5C451ED648583722F5834fF6490D4570f7d</span></p>
    <p><strong>Bitcoin (BTC):</strong><br><span class="crypto-address">38JkmWhTFYosecu45ewoheYMjJw68sHSj3</span></p>
    
    <p><small>Make sure to send USDT on the Ethereum network (ERC-20). Do not send other tokens or cryptocurrencies to these addresses.</small></p>
</body>
</html>`)
    })

    // Endpoint per cancellare la chat
    http.HandleFunc("/clear", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost {
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
            return
        }
        sessionID, err := r.Cookie("session_id")
        if err != nil {
            http.Error(w, "Error: Session not found", http.StatusBadRequest)
            return
        }
        mutex.Lock()
        delete(sessions, sessionID.Value)
        mutex.Unlock()
        w.WriteHeader(http.StatusOK)
    })

    // Endpoint /ask con sintesi personalizzata per stile ARCA-b e Grok
    http.HandleFunc("/ask", func(w http.ResponseWriter, r *http.Request) {
        fmt.Println("Ricevuta richiesta su /ask")
        sessionID, err := r.Cookie("session_id")
        if err != nil {
            http.Error(w, "Error: Session not found", http.StatusBadRequest)
            return
        }

        question := r.URL.Query().Get("question")
        if question == "" {
            http.Error(w, "Error: Please specify a question with ?question=", http.StatusBadRequest)
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
            openAIAnswer = "Error: OPENAI_API_KEY is not set."
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
                fmt.Printf("Error with OpenAI: %v\n", err)
                openAIAnswer = "Error: OpenAI did not respond."
            } else {
                openAIAnswer = openAIResp.Choices[0].Message.Content
            }
        }

        // 2. DeepSeek
        var deepSeekAnswer string
        deepSeekAnswer, err = getDeepSeekResponse(session.History)
        if err != nil {
            fmt.Printf("Error with DeepSeek: %v\n", err)
            deepSeekAnswer = "Error: DeepSeek did not respond."
        }

        // 3. Gemini
        var geminiAnswer string
        if geminiKey == "" {
            geminiAnswer = "Error: GEMINI_API_KEY is not set."
        } else {
            historyForGemini := ""
            for _, msg := range session.History {
                historyForGemini += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
            }
            geminiReq, err := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key="+geminiKey,
                strings.NewReader(fmt.Sprintf(`{"contents":[{"parts":[{"text":"%s"}]}]}`, historyForGemini)))
            if err != nil {
                fmt.Printf("Error creating request to Gemini: %v\n", err)
                geminiAnswer = "Error: Gemini did not respond."
            } else {
                geminiReq.Header.Set("Content-Type", "application/json")
                geminiResp, err := client.Do(geminiReq)
                if err != nil {
                    fmt.Printf("Error with Gemini: %v\n", err)
                    geminiAnswer = "Error: Gemini did not respond."
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
                        fmt.Printf("Error parsing Gemini response: %v\n", err)
                        geminiAnswer = "Error: Gemini did not respond correctly."
                    } else if len(geminiResult.Candidates) == 0 || len(geminiResult.Candidates[0].Content.Parts) == 0 {
                        geminiAnswer = "Error: No valid response from Gemini."
                    } else {
                        geminiAnswer = geminiResult.Candidates[0].Content.Parts[0].Text
                    }
                }
            }
        }

        // 4. Costruisci rawResponses e synthesisParts senza filtro censura
        rawResponses := strings.Builder{}
        rawResponses.WriteString("### Original Responses\n\n")
        synthesisParts := []string{
            fmt.Sprintf("OpenAI: %s", openAIAnswer),
            fmt.Sprintf("DeepSeek: %s", deepSeekAnswer),
            fmt.Sprintf("Gemini: %s", geminiAnswer),
        }
        rawResponses.WriteString("#### OpenAI\n" + openAIAnswer + "\n\n")
        rawResponses.WriteString("#### DeepSeek\n" + deepSeekAnswer + "\n\n")
        rawResponses.WriteString("#### Gemini\n" + geminiAnswer + "\n\n")

        // 5. Sintesi con DeepInfra, AIMLAPI, Hugging Face
        var synthesizedAnswer string
        if len(synthesisParts) == 0 {
            synthesizedAnswer = "Error: No valid responses to synthesize."
        } else {
            // Prompt personalizzato per stile ARCA-b o Grok
            var synthesisPrompt string
            if style == "arca-b" {
                synthesisPrompt = fmt.Sprintf(
                    "L'utente ha chiesto: '%s'. Mescola queste risposte in una sola, bella chiara, che risponde dritto al punto. Tira fuori il meglio da ognuna, buttando dentro un po’ di scienza, cultura, storia e tecnologia, ma fallo in stile ARCA-b: semplice, informale, come se lo spiegassi a un amico davanti a una birra. Niente paroloni, solo roba interessante:\n\n%s",
                    question, strings.Join(synthesisParts, "\n\n"),
                )
            } else { // stile grok o default
                synthesisPrompt = fmt.Sprintf(
                    "The user asked: '%s'. Synthesize the following answers into a single, comprehensive response that directly addresses the user's question. Integrate the most interesting aspects from each response, combining scientific, cultural, historical, and technological perspectives to reflect a global digital knowledge. Provide a clear and concise answer in a formal, detailed style:\n\n%s",
                    question, strings.Join(synthesisParts, "\n\n"),
                )
            }

            // Prova DeepInfra per la sintesi
            synthesizedAnswer, err = getDeepInfraResponse(deepInfraKey, client, synthesisPrompt)
            if err != nil {
                fmt.Printf("Error synthesizing with DeepInfra: %v\n", err)
                // Prova AIMLAPI come fallback
                synthesizedAnswer, err = getAIMLAPIResponse(aimlKey, client, synthesisPrompt)
                if err != nil {
                    fmt.Printf("Error synthesizing with AIMLAPI: %v\n", err)
                    // Prova Hugging Face come ulteriore fallback
                    synthesizedAnswer, err = getHuggingFaceResponse(huggingFaceKey, client, synthesisPrompt)
                    if err != nil {
                        fmt.Printf("Error synthesizing with Hugging Face: %v\n", err)
                        // Fallback finale: usa la prima risposta valida disponibile
                        if !strings.Contains(openAIAnswer, "Error") {
                            synthesizedAnswer = openAIAnswer + " (Note: Synthesis not available, using OpenAI response.)"
                        } else if !strings.Contains(geminiAnswer, "Error") {
                            synthesizedAnswer = geminiAnswer + " (Note: Synthesis not available, using Gemini response.)"
                        } else if !strings.Contains(deepSeekAnswer, "Error") {
                            synthesizedAnswer = deepSeekAnswer + " (Note: Synthesis not available, using DeepSeek response.)"
                        } else {
                            synthesizedAnswer = "Error: I couldn't synthesize the responses."
                        }
                    }
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
            fmt.Printf("Error sending JSON response: %v\n", err)
            http.Error(w, "Internal server error", http.StatusInternalServerError)
            return
        }
    })

    fmt.Printf("Server in ascolto sulla porta %s...\n", port)
    if err := http.ListenAndServe(":"+port, nil); err != nil {
        fmt.Printf("Errore nell'avvio del server: %v\n", err)
        os.Exit(1)
    }
}

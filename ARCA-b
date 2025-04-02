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

type UserRequestTracker struct {
    HourlyCount   int       // Contatore orario
    LastResetHour time.Time // Ultimo reset del contatore orario
    IsPremium     bool      // Flag per utenti premium
}

var (
    sessions        = make(map[string]*Session)
    requestTrackers = make(map[string]*UserRequestTracker)
    premiumUsers    = make(map[string]bool) // Lista di session_ids premium
    mutex           = &sync.Mutex{}
    hourlyLimit     = 15                    // Limite orario per utenti non premium
)

// getDeepInfraResponse to synthesize responses using the DeepInfra API
func getDeepInfraResponse(deepInfraKey string, client *http.Client, prompt string) (string, error) {
    if deepInfraKey == "" {
        return "", fmt.Errorf("DEEPINFRA_API_KEY is not set")
    }

    prompt = strings.ReplaceAll(prompt, "\n", " ")
    prompt = strings.ReplaceAll(prompt, "\"", "\\\"")
    logLimit := 100
    if len(prompt) < logLimit {
        logLimit = len(prompt)
    }
    fmt.Println("Sending request to DeepInfra with prompt (first 100 chars):", prompt[:logLimit], "...")

    payload := fmt.Sprintf(`{"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "user", "content": "%s"}], "max_tokens": 1000, "temperature": 0.7}`, prompt)
    req, err := http.NewRequest("POST", "https://api.deepinfra.com/v1/openai/chat/completions", strings.NewReader(payload))
    if err != nil {
        return "", fmt.Errorf("error creating request to DeepInfra: %v", err)
    }

    req.Header.Set("Authorization", "Bearer "+deepInfraKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")

    var resp *http.Response
    for attempt := 1; attempt <= 3; attempt++ {
        resp, err = client.Do(req)
        if err == nil {
            break
        }
        fmt.Printf("Error with DeepInfra (attempt %d): %v\n", attempt, err)
        time.Sleep(time.Second * time.Duration(attempt))
    }
    if err != nil {
        return "", fmt.Errorf("error with DeepInfra after 3 attempts: %v", err)
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", fmt.Errorf("error reading DeepInfra response: %v", err)
    }
    fmt.Println("Raw response from DeepInfra (status %d): %s", resp.StatusCode, string(body))

    var deepInfraResult struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
        Error string `json:"error"`
    }
    if err := json.Unmarshal(body, &deepInfraResult); err != nil {
        return "", fmt.Errorf("error parsing DeepInfra response: %v, raw response: %s", err, string(body))
    }
    if deepInfraResult.Error != "" {
        return "", fmt.Errorf("error from DeepInfra: %s", deepInfraResult.Error)
    }
    if len(deepInfraResult.Choices) == 0 || deepInfraResult.Choices[0].Message.Content == "" {
        return "", fmt.Errorf("no valid response from DeepInfra: %s", string(body))
    }

    return deepInfraResult.Choices[0].Message.Content, nil
}

// getAIMLAPIResponse to synthesize responses using the AIMLAPI API
func getAIMLAPIResponse(aimlKey string, client *http.Client, prompt string) (string, error) {
    if aimlKey == "" {
        return "", fmt.Errorf("AIMLAPI_API_KEY is not set")
    }

    prompt = strings.ReplaceAll(prompt, "\n", " ")
    prompt = strings.ReplaceAll(prompt, "\"", "\\\"")
    logLimit := 100
    if len(prompt) < logLimit {
        logLimit = len(prompt)
    }
    fmt.Println("Sending request to AIMLAPI with prompt (first 100 chars):", prompt[:logLimit], "...")

    payload := fmt.Sprintf(`{"model": "Grok", "messages": [{"role": "user", "content": "%s"}]}`, prompt)
    req, err := http.NewRequest("POST", "https://api.aimlapi.com/v1/chat/completions", strings.NewReader(payload))
    if err != nil {
        return "", fmt.Errorf("error creating request to AIMLAPI: %v", err)
    }

    req.Header.Set("Authorization", "Bearer "+aimlKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")

    var resp *http.Response
    for attempt := 1; attempt <= 3; attempt++ {
        resp, err = client.Do(req)
        if err == nil {
            break
        }
        fmt.Printf("Error with AIMLAPI (attempt %d): %v\n", attempt, err)
        time.Sleep(time.Second * time.Duration(attempt))
    }
    if err != nil {
        return "", fmt.Errorf("error with AIMLAPI after 3 attempts: %v", err)
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", fmt.Errorf("error reading AIMLAPI response: %v", err)
    }
    fmt.Println("Raw response from AIMLAPI (status %d): %s", resp.StatusCode, string(body))

    var aimlResult struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
        Error string `json:"error"`
    }
    if err := json.Unmarshal(body, &aimlResult); err != nil {
        return "", fmt.Errorf("error parsing AIMLAPI response: %v, raw response: %s", err, string(body))
    }
    if aimlResult.Error != "" {
        return "", fmt.Errorf("error from AIMLAPI: %s", aimlResult.Error)
    }
    if len(aimlResult.Choices) == 0 || aimlResult.Choices[0].Message.Content == "" {
        return "", fmt.Errorf("no valid response from AIMLAPI: %s", string(body))
    }

    return aimlResult.Choices[0].Message.Content, nil
}

// getHuggingFaceResponse to synthesize responses using the Hugging Face API
func getHuggingFaceResponse(hfKey string, client *http.Client, prompt string) (string, error) {
    if hfKey == "" {
        return "", fmt.Errorf("HUGGINGFACE_API_KEY is not set")
    }

    prompt = strings.ReplaceAll(prompt, "\n", " ")
    prompt = strings.ReplaceAll(prompt, "\"", "\\\"")
    logLimit := 100
    if len(prompt) < logLimit {
        logLimit = len(prompt)
    }
    fmt.Println("Sending request to Hugging Face with prompt (first 100 chars):", prompt[:logLimit], "...")

    payload := fmt.Sprintf(`{"inputs": "%s", "parameters": {"max_length": 500, "temperature": 0.7, "top_p": 0.9}}`, prompt)
    req, err := http.NewRequest("POST", "https://api-inference.huggingface.co/models/distilgpt2", strings.NewReader(payload))
    if err != nil {
        return "", fmt.Errorf("error creating request to Hugging Face: %v", err)
    }

    req.Header.Set("Authorization", "Bearer "+hfKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")

    var resp *http.Response
    for attempt := 1; attempt <= 3; attempt++ {
        resp, err = client.Do(req)
        if err == nil {
            break
        }
        fmt.Printf("Error with Hugging Face (attempt %d): %v\n", attempt, err)
        time.Sleep(time.Second * time.Duration(attempt))
    }
    if err != nil {
        return "", fmt.Errorf("error with Hugging Face after 3 attempts: %v", err)
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", fmt.Errorf("error reading Hugging Face response: %v", err)
    }
    fmt.Println("Raw response from Hugging Face (status %d): %s", resp.StatusCode, string(body))

    var hfResult []struct {
        GeneratedText string `json:"generated_text"`
    }
    if err := json.Unmarshal(body, &hfResult); err != nil {
        var errorResult struct {
            Error string `json:"error"`
        }
        if json.Unmarshal(body, &errorResult) == nil && errorResult.Error != "" {
            return "", fmt.Errorf("error from Hugging Face: %s", errorResult.Error)
        }
        return "", fmt.Errorf("error parsing Hugging Face response: %v, raw response: %s", err, string(body))
    }
    if len(hfResult) == 0 || hfResult[0].GeneratedText == "" {
        return "", fmt.Errorf("no valid response from Hugging Face: %s", string(body))
    }

    generatedText := strings.TrimPrefix(hfResult[0].GeneratedText, prompt)
    return strings.TrimSpace(generatedText), nil
}

// getMistralResponse to get responses using the Mistral API
func getMistralResponse(mistralKey string, client *http.Client, prompt string) (string, error) {
    if mistralKey == "" {
        return "", fmt.Errorf("MISTRAL_API_KEY is not set")
    }

    prompt = strings.ReplaceAll(prompt, "\n", " ")
    prompt = strings.ReplaceAll(prompt, "\"", "\\\"")
    logLimit := 100
    if len(prompt) < logLimit {
        logLimit = len(prompt)
    }
    fmt.Println("Sending request to Mistral with prompt (first 100 chars):", prompt[:logLimit], "...")

    payload := fmt.Sprintf(`{"model": "mistral-small-latest", "messages": [{"role": "user", "content": "%s"}], "max_tokens": 1000, "temperature": 0.7}`, prompt)
    req, err := http.NewRequest("POST", "https://api.mistral.ai/v1/chat/completions", strings.NewReader(payload))
    if err != nil {
        return "", fmt.Errorf("error creating request to Mistral: %v", err)
    }

    req.Header.Set("Authorization", "Bearer "+mistralKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")

    var resp *http.Response
    for attempt := 1; attempt <= 3; attempt++ {
        resp, err = client.Do(req)
        if err == nil {
            break
        }
        fmt.Printf("Error with Mistral (attempt %d): %v\n", attempt, err)
        time.Sleep(time.Second * time.Duration(attempt))
    }
    if err != nil {
        return "", fmt.Errorf("error with Mistral after 3 attempts: %v", err)
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", fmt.Errorf("error reading Mistral response: %v", err)
    }
    fmt.Println("Raw response from Mistral (status %d): %s", resp.StatusCode, string(body))

    var mistralResult struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
        Error string `json:"error"`
    }
    if err := json.Unmarshal(body, &mistralResult); err != nil {
        return "", fmt.Errorf("error parsing Mistral response: %v, raw response: %s", err, string(body))
    }
    if mistralResult.Error != "" {
        return "", fmt.Errorf("error from Mistral: %s", mistralResult.Error)
    }
    if len(mistralResult.Choices) == 0 || mistralResult.Choices[0].Message.Content == "" {
        return "", fmt.Errorf("no valid response from Mistral: %s", string(body))
    }

    return mistralResult.Choices[0].Message.Content, nil
}

func getDeepSeekResponse(client *http.Client, deepSeekKey string, messages []openai.ChatCompletionMessage, language string) (string, error) {
    if deepSeekKey == "" {
        return "", fmt.Errorf("DEEPSEEK_API_KEY is not set")
    }
    var deepSeekMessages []map[string]string
    for i, msg := range messages {
        if i == len(messages)-1 { // Ultimo messaggio (la domanda)
            deepSeekMessages = append(deepSeekMessages, map[string]string{
                "role":    msg.Role,
                "content": fmt.Sprintf("Respond in %s: %s", language, msg.Content),
            })
        } else {
            deepSeekMessages = append(deepSeekMessages, map[string]string{
                "role":    msg.Role,
                "content": msg.Content,
            })
        }
    }
    body, err := json.Marshal(map[string]interface{}{
        "model":    "deepseek-chat",
        "messages": deepSeekMessages,
    })
    if err != nil {
        return "", fmt.Errorf("error creating JSON body: %v", err)
    }
    req, err := http.NewRequest("POST", "https://api.deepseek.com/v1/chat/completions", strings.NewReader(string(body)))
    if err != nil {
        return "", fmt.Errorf("error creating request: %v", err)
    }
    req.Header.Set("Authorization", "Bearer "+deepSeekKey)
    req.Header.Set("Content-Type", "application/json")
    resp, err := client.Do(req)
    if err != nil {
        return "", fmt.Errorf("error requesting DeepSeek: %v", err)
    }
    defer resp.Body.Close()
    bodyResp, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", fmt.Errorf("error reading response: %v", err)
    }
    if resp.StatusCode != http.StatusOK {
        return "", fmt.Errorf("invalid response from DeepSeek (status %d): %s", resp.StatusCode, string(bodyResp))
    }
    var result struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
    }
    if err := json.Unmarshal(bodyResp, &result); err != nil {
        return "", fmt.Errorf("error parsing JSON: %v, raw response: %s", err, string(bodyResp))
    }
    if len(result.Choices) > 0 {
        return result.Choices[0].Message.Content, nil
    }
    return "", fmt.Errorf("no valid response from DeepSeek: %s", string(bodyResp))
}
func main() {
    openAIKey := os.Getenv("OPENAI_API_KEY")
    deepSeekKey := os.Getenv("DEEPSEEK_API_KEY")
    geminiKey := os.Getenv("GEMINI_API_KEY")
    deepInfraKey := os.Getenv("DEEPINFRA_API_KEY")
    aimlKey := os.Getenv("AIMLAPI_API_KEY")
    huggingFaceKey := os.Getenv("HUGGINGFACE_API_KEY")
    mistralKey := os.Getenv("MISTRAL_API_KEY")

    fmt.Printf("Loading API keys for arcab...\n")
    if openAIKey == "" {
        fmt.Println("Error: OPENAI_API_KEY is not set")
    } else {
        fmt.Println("OPENAI_API_KEY loaded successfully")
    }
    if deepSeekKey == "" {
        fmt.Println("Error: DEEPSEEK_API_KEY is not set")
    } else {
        fmt.Println("DEEPSEEK_API_KEY loaded successfully")
    }
    if geminiKey == "" {
        fmt.Println("Error: GEMINI_API_KEY is not set")
    } else {
        fmt.Println("GEMINI_API_KEY loaded successfully")
    }
    if deepInfraKey == "" {
        fmt.Println("Error: DEEPINFRA_API_KEY is not set")
    } else {
        fmt.Println("DEEPINFRA_API_KEY loaded successfully")
    }
    if aimlKey == "" {
        fmt.Println("Error: AIMLAPI_API_KEY is not set")
    } else {
        fmt.Println("AIMLAPI_API_KEY loaded successfully")
    }
    if huggingFaceKey == "" {
        fmt.Println("Error: HUGGINGFACE_API_KEY is not set")
    } else {
        fmt.Println("HUGGINGFACE_API_KEY loaded successfully")
    }
    if mistralKey == "" {
        fmt.Println("Error: MISTRAL_API_KEY is not set")
    } else {
        fmt.Println("MISTRAL_API_KEY loaded successfully")
    }

    port := os.Getenv("PORT")
    if port == "" {
        fmt.Println("PORT not specified, using default :8080")
        port = "8080"
    } else {
        fmt.Printf("PORT specified from environment: %s\n", port)
    }

    openAIClient := openai.NewClient(openAIKey)
    client := &http.Client{
        Timeout: 30 * time.Second,
    }

    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        fmt.Println("Received request on /health")
        w.WriteHeader(http.StatusOK)
        fmt.Fprint(w, "arcab Chat AI is running on arcab-global-ai.org")
    })

    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Println("Received request on /")
        sessionID, err := r.Cookie("session_id")
        if err != nil || sessionID == nil {
            sessionID = &http.Cookie{
                Name:  "session_id",
                Value: uuid.New().String(),
                Path:  "/",
            }
            http.SetCookie(w, sessionID)
        }
        w.Header().Set("Content-Type", "text/html; charset=utf-8")
        fmt.Fprintf(w, `
<!DOCTYPE html>
<html>
<head>
    <title>arcab Chat AI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
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
            margin-bottom: 15px;
            text-align: center;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 20px;
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
            position: relative;
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
        .share-button {
            padding: 5px 10px;
            font-size: 0.8em;
            margin-left: 10px;
            background-color: #28a745;
        }
        .share-button:hover {
            background-color: #218838;
        }
        body.dark .share-button {
            background-color: #2ecc71;
        }
        body.dark .share-button:hover {
            background-color: #27ae60;
        }
        .processing {
            font-style: italic;
            color: #666;
            margin: 5px 0;
            text-align: left;
        }
        body.dark .processing {
            color: #aaa;
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
        .vision-text {
            text-align: center;
            font-size: 0.9em;
            color: #666;
            margin-bottom: 20px;
            line-height: 1.4;
        }
        body.dark .vision-text {
            color: #aaa;
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
                gap: 10px;
                align-items: center;
            }
            .button-container button {
                width: 100%;
                max-width: 200px;
            }
            .share-button {
                margin-left: 0;
                margin-top: 5px;
            }
        }
    </style>
</head>
<body>
    <h1>arcab Chat AI</h1>
    <p style="text-align: center; font-size: 0.9em; color: #666; margin-bottom: 10px;">
        Note: Conversations are temporarily saved to improve the experience. Do not enter personal or sensitive information.
    </p>
    <p class="vision-text">
        <strong>Vision:</strong> arcab Chat AI aims to unleash the full power of global digital knowledge for everyone, tapping into every AI out there to gather all available data - something no single AI can do alone. It delivers uncensored, propaganda-free answers by blending the best insights from every source into one ultimate response. As an open-source project, arcab is built for scalability and is oriented towards a P2P future, empowering communities to collaborate and grow together.
    </p>
    <div class="button-container">
        <button onclick="toggleTheme()">Dark/Light Theme</button>
        <a href="/donate"><button>Donate</button></a>
        <a href="mailto:arcab.founder@gmail.com"><button>Contact</button></a>
    </div>
    <div id="chat-container">
        <div id="chat"></div>
    </div>
    <div class="input-and-style-container">
        <div class="style-container">
            <select id="style-select">
                <option value="grok">Informal response style</option>
                <option value="arca-b">arcab response style</option>
            </select>
            <select id="language-select">
                <option value="Italiano">Italiano</option>
                <option value="English">English</option>
                <option value="Deutsch">Deutsch</option>
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
        const languageSelect = document.getElementById("language-select");

        if (localStorage.getItem("theme") === "dark") {
            document.body.classList.add("dark");
        }
        if (localStorage.getItem("style") === "arca-b") {
            styleSelect.value = "arca-b";
        } else {
            styleSelect.value = "grok";
        }
        if (localStorage.getItem("language")) {
            languageSelect.value = localStorage.getItem("language");
        } else {
            languageSelect.value = "Italiano"; // Default
        }

        function toggleTheme() {
            document.body.classList.toggle("dark");
            localStorage.setItem("theme", document.body.classList.contains("dark") ? "dark" : "light");
        }

        function addMessage(text, isUser, style, rawResponses) {
            const div = document.createElement("div");
            const messageText = (isUser ? "You: " : "arcab: ") + text;
            div.innerHTML = messageText.replace(/\n/g, "<br>");
            div.className = "message " + (isUser ? "user" : "bot");
            chat.appendChild(div);

            if (!isUser) {
                const shareButton = document.createElement("button");
                shareButton.textContent = "Share";
                shareButton.className = "share-button";
                shareButton.onclick = () => shareResponse(getLastUserQuestion(), text);
                div.appendChild(shareButton);
            }

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

        function getLastUserQuestion() {
            const messages = chat.getElementsByClassName("message user");
            if (messages.length > 0) {
                const lastMessage = messages[messages.length - 1];
                return lastMessage.textContent.replace("You: ", "").trim();
            }
            return "No question found";
        }

        function shareResponse(question, answer) {
            const shareText = "Question: " + question + "\nAnswer from arcab Chat AI: " + answer + "\n\nTry it yourself at: https://arcab-global-ai.org";
            if (navigator.share) {
                navigator.share({
                    title: "arcab Chat AI Response",
                    text: shareText,
                    url: "https://arcab-global-ai.org",
                }).catch(err => {
                    console.error("Error sharing:", err);
                    fallbackCopyToClipboard(shareText);
                });
            } else {
                fallbackCopyToClipboard(shareText);
            }
        }

        function fallbackCopyToClipboard(text) {
            navigator.clipboard.writeText(text).then(() => {
                alert("Response copied to clipboard!");
            }).catch(err => {
                console.error("Failed to copy to clipboard:", err);
                alert("Failed to copy to clipboard. Please copy manually:\n\n" + text);
            });
        }

        function showProcessingMessage() {
            const div = document.createElement("div");
            div.id = "processing-message";
            div.className = "processing";
            div.textContent = "Response being processed";
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        function removeProcessingMessage() {
            const processingMessage = document.getElementById("processing-message");
            if (processingMessage) {
                processingMessage.remove();
            }
        }

        async function sendMessage() {
            const question = input.value.trim();
            if (!question) return;
            addMessage(question, true);
            input.value = "";

            const style = styleSelect.value;
            const language = languageSelect.value;
            localStorage.setItem("style", style);
            localStorage.setItem("language", language);

            showProcessingMessage();
            try {
                const minDisplayTime = new Promise(resolve => setTimeout(resolve, 1000));
                const response = await fetch("/ask?question=" + encodeURIComponent(question) + "&style=" + style + "&language=" + encodeURIComponent(language), {
                    credentials: "include"
                });
                const answer = await Promise.all([response.json(), minDisplayTime]);
                console.log("Response received from /ask:", answer[0]);
                removeProcessingMessage();
                const rawResponses = answer[0].rawResponses || "";
                const synthesizedAnswer = answer[0].synthesized || "Error: No synthesized response.";
                addMessage(synthesizedAnswer, false, style, rawResponses);
            } catch (error) {
                console.error("Error during request:", error);
                removeProcessingMessage();
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

    http.HandleFunc("/donate", func(w http.ResponseWriter, r *http.Request) {
        fmt.Println("Received request on /donate")
        w.Header().Set("Content-Type", "text/html; charset=utf-8")
        fmt.Fprintf(w, `<!DOCTYPE html>
<html>
<head>
    <title>Donations - arcab Chat AI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; text-align: center; }
        button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1em; margin: 10px; }
        button:hover { background-color: #0056b3; }
        code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; }
        .crypto-address { margin: 10px 0; word-wrap: break-word; }
    </style>
</head>
<body>
    <h1>Support arcab Chat AI</h1>
    <p>Your donations help us improve the project and keep it free from censorship and propaganda. Thank you!</p>
    <p>Please send your donation to one of the following cryptocurrency addresses:</p>
    <div class="crypto-address"><strong>Bitcoin (BTC):</strong> <code>38JkmWhTFYosecu45ewoheYMjJw68sHSj3</code></div>
    <div class="crypto-address"><strong>USDT (Ethereum):</strong> <code>0x71ECB5C451ED648583722F5834fF6490D4570f7d</code></div>
    <p><small>After donating, contact us at <a href="mailto:arcab.founder@gmail.com">arcab.founder@gmail.com</a> with your session ID to unlock premium features.</small></p>
    <a href="/"><button>Back to Chat</button></a>
</body>
</html>`)
    })

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

    http.HandleFunc("/ask", func(w http.ResponseWriter, r *http.Request) {
        fmt.Println("Received request on /ask")
        sessionID, err := r.Cookie("session_id")
        if err != nil {
            http.Error(w, "Error: Session not found", http.StatusBadRequest)
            return
        }

        mutex.Lock()
        tracker, exists := requestTrackers[sessionID.Value]
        if !exists {
            tracker = &UserRequestTracker{
                HourlyCount:   0,
                LastResetHour: time.Now(),
                IsPremium:     premiumUsers[sessionID.Value],
            }
            requestTrackers[sessionID.Value] = tracker
        }

        if !tracker.IsPremium {
            if time.Since(tracker.LastResetHour) > time.Hour {
                tracker.HourlyCount = 0
                tracker.LastResetHour = time.Now()
            }
            tracker.HourlyCount++
            fmt.Printf("User %s: %d requests this hour\n", sessionID.Value, tracker.HourlyCount)
            if tracker.HourlyCount > hourlyLimit {
                mutex.Unlock()
                response := map[string]string{
                    "synthesized":  "Hai raggiunto il limite orario di 15 domande. Considera di supportarci con una donazione per mantenere il progetto attivo! Visita la pagina <a href=\"/donate\">Dona</a>.",
                    "rawResponses": "",
                }
                w.Header().Set("Content-Type", "application/json")
                json.NewEncoder(w).Encode(response)
                return
            }
        }
        mutex.Unlock()

        question := r.URL.Query().Get("question")
        if question == "" {
            http.Error(w, "Errore: Specifica una domanda con ?question=", http.StatusBadRequest)
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

        language := r.URL.Query().Get("language")
        if language == "" {
            language = "Italiano" // Default
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

        // Struttura per raccogliere le risposte
        type aiResponse struct {
            name    string
            content string
            err     error
        }

        // Canale per raccogliere le risposte
        responses := make(chan aiResponse, 4)
        var wg sync.WaitGroup

        // OpenAI
        wg.Add(1)
        go func() {
            defer wg.Done()
            var answer string
            if openAIKey == "" {
                answer = fmt.Sprintf("Errore: OPENAI_API_KEY non è impostata. (in %s)", language)
            } else {
                ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
                defer cancel()
                messagesWithLang := append([]openai.ChatCompletionMessage{}, session.History...)
                messagesWithLang[len(messagesWithLang)-1].Content = fmt.Sprintf("Respond in %s: %s", language, question)
                resp, err := openAIClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
                    Model:    openai.GPT3Dot5Turbo,
                    Messages: messagesWithLang,
                })
                if err != nil {
                    fmt.Printf("Errore con OpenAI: %v\n", err)
                    answer = fmt.Sprintf("Errore: OpenAI non ha risposto. (in %s)", language)
                } else {
                    answer = resp.Choices[0].Message.Content
                }
            }
            responses <- aiResponse{name: "OpenAI", content: answer, err: nil}
        }()

        // DeepSeek
        wg.Add(1)
        go func() {
            defer wg.Done()
            answer, err := getDeepSeekResponse(client, deepSeekKey, session.History, language)
            if err != nil {
                fmt.Printf("Errore con DeepSeek: %v\n", err)
                answer = fmt.Sprintf("Errore: DeepSeek non ha risposto. (in %s)", language)
            }
            responses <- aiResponse{name: "DeepSeek", content: answer, err: err}
        }()

        // Gemini
        wg.Add(1)
        go func() {
            defer wg.Done()
            var answer string
            if geminiKey == "" {
                answer = fmt.Sprintf("Errore: GEMINI_API_KEY non è impostata. (in %s)", language)
            } else {
                historyForGemini := ""
                for i, msg := range session.History {
                    if i == len(session.History)-1 {
                        historyForGemini += fmt.Sprintf("%s: Respond in %s: %s\n", msg.Role, language, msg.Content)
                    } else {
                        historyForGemini += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
                    }
                }
                req, err := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key="+geminiKey,
                    strings.NewReader(fmt.Sprintf(`{"contents":[{"parts":[{"text":"%s"}]}]}`, historyForGemini)))
                if err != nil {
                    fmt.Printf("Errore nella creazione della richiesta a Gemini: %v\n", err)
                    answer = fmt.Sprintf("Errore: Gemini non ha risposto. (in %s)", language)
                } else {
                    req.Header.Set("Content-Type", "application/json")
                    resp, err := client.Do(req)
                    if err != nil {
                        fmt.Printf("Errore con Gemini: %v\n", err)
                        answer = fmt.Sprintf("Errore: Gemini non ha risposto. (in %s)", language)
                    } else {
                        defer resp.Body.Close()
                        var result struct {
                            Candidates []struct {
                                Content struct {
                                    Parts []struct {
                                        Text string `json:"text"`
                                    } `json:"parts"`
                                } `json:"content"`
                            } `json:"candidates"`
                        }
                        if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
                            fmt.Printf("Errore nel parsing della risposta di Gemini: %v\n", err)
                            answer = fmt.Sprintf("Errore: Gemini non ha risposto correttamente. (in %s)", language)
                        } else if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
                            answer = fmt.Sprintf("Errore: Nessuna risposta valida da Gemini. (in %s)", language)
                        } else {
                            answer = result.Candidates[0].Content.Parts[0].Text
                        }
                    }
                }
            }
            responses <- aiResponse{name: "Gemini", content: answer, err: nil}
        }()

        // Mistral
        wg.Add(1)
        go func() {
            defer wg.Done()
            prompt := fmt.Sprintf("Respond in %s: %s", language, question)
            answer, err := getMistralResponse(mistralKey, client, prompt)
            if err != nil {
                fmt.Printf("Errore con Mistral: %v\n", err)
                answer = fmt.Sprintf("Errore: Mistral non ha risposto. (in %s)", language)
            }
            responses <- aiResponse{name: "Mistral", content: answer, err: err}
        }()

        // Chiudi il canale quando tutte le goroutine sono terminate
        go func() {
            wg.Wait()
            close(responses)
        }()

        // Raccogli le risposte
        rawResponses := strings.Builder{}
        rawResponses.WriteString("### Original Responses\n\n")
        synthesisParts := make([]string, 0, 4)
        for resp := range responses {
            synthesisParts = append(synthesisParts, fmt.Sprintf("%s: %s", resp.name, resp.content))
            rawResponses.WriteString(fmt.Sprintf("#### %s\n%s\n\n", resp.name, resp.content))
        }

        // Sintesi delle risposte
        var synthesizedAnswer string
        if len(synthesisParts) == 0 {
            synthesizedAnswer = fmt.Sprintf("Error: No valid responses to synthesize. (in %s)", language)
        } else {
            var synthesisPrompt string
            if style == "arca-b" {
                synthesisPrompt = fmt.Sprintf(
                    "Respond in %s: Take these responses and mix them into one, using the best from each, with a bit of science, culture, history, and tech, in a simple and informal style like you're talking to a friend over a beer. No introductions, straight to the point:\n\n%s",
                    language, strings.Join(synthesisParts, "\n\n"),
                )
            } else {
                synthesisPrompt = fmt.Sprintf(
                    "Respond in %s: The user asked: '%s'. Synthesize these responses into a complete and direct answer, blending the most interesting aspects of each, with scientific, cultural, historical, and technological perspectives, in an informal and clear style:\n\n%s",
                    language, question, strings.Join(synthesisParts, "\n\n"),
                )
            }

            synthesizedAnswer, err = getDeepInfraResponse(deepInfraKey, client, synthesisPrompt)
            if err != nil {
                fmt.Printf("Errore nella sintesi con DeepInfra: %v\n", err)
                synthesizedAnswer, err = getAIMLAPIResponse(aimlKey, client, synthesisPrompt)
                if err != nil {
                    fmt.Printf("Errore nella sintesi con AIMLAPI: %v\n", err)
                    synthesizedAnswer, err = getHuggingFaceResponse(huggingFaceKey, client, synthesisPrompt)
                    if err != nil {
                        fmt.Printf("Errore nella sintesi con Hugging Face: %v\n", err)
                        for _, part := range synthesisParts {
                            if !strings.Contains(part, "Errore") {
                                synthesizedAnswer = strings.TrimPrefix(part, part[:strings.Index(part, ":")+2]) + fmt.Sprintf(" (Note: Synthesis not available, using %s response, in %s)", strings.Split(part, ":")[0], language)
                                break
                            }
                        }
                        if synthesizedAnswer == "" {
                            synthesizedAnswer = fmt.Sprintf("Errore: Non sono riuscito a sintetizzare le risposte. (in %s)", language)
                        }
                    }
                }
            }
        }

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

    fmt.Printf("arcab server in ascolto sulla porta %s...\n", port)
    if err := http.ListenAndServe(":"+port, nil); err != nil {
        fmt.Printf("Errore nell'avvio del server: %v\n", err)
        os.Exit(1)
    }
}

package main

import (
    "context"
    "encoding/json"
    "fmt"
    "io"
    "math"
    "net/http"
    "os"
    "strings"
    "sync"
    "time"

    "github.com/google/uuid"
    "github.com/sashabaranov/go-openai"
)

// --- SEZIONE 1: Strutture dati e variabili globali ---
type Session struct {
    History []openai.ChatCompletionMessage
}

type UserRequestTracker struct {
    HourlyCount   int
    LastResetHour time.Time
    IsPremium     bool
}

type ChatRequest struct {
    Message   string `json:"message"` // Messaggio in chiaro
    Style     string `json:"style"`
    Language  string `json:"language"`
}

type ChatResponse struct {
    Response      string `json:"response"` // Risposta in chiaro
    RawResponses  string `json:"rawResponses"`
    Contributions string `json:"contributions"`
}

var (
    sessions        = make(map[string]*Session)
    requestTrackers = make(map[string]*UserRequestTracker)
    premiumUsers    = make(map[string]bool)
    conversations   = make(map[string]ChatResponse) // Nuova mappa per salvare le conversazioni
    mutex           = &sync.Mutex{}
    hourlyLimit     = 15
)

// --- SEZIONE 2: Funzioni per le API (DeepInfra, AIMLAPI, HuggingFace, Mistral, DeepSeek, Cohere) ---
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

func getCohereResponse(cohereKey string, client *http.Client, prompt string) (string, error) {
    if cohereKey == "" {
        return "", fmt.Errorf("COHERE_API_KEY is not set")
    }

    prompt = strings.ReplaceAll(prompt, "\n", " ")
    prompt = strings.ReplaceAll(prompt, "\"", "\\\"")
    logLimit := 100
    if len(prompt) < logLimit {
        logLimit = len(prompt)
    }
    fmt.Println("Sending request to Cohere with prompt (first 100 chars):", prompt[:logLimit], "...")

    fullPrompt := fmt.Sprintf("Rispondi esclusivamente in italiano. Non usare altre lingue, nemmeno per frasi brevi o parole singole. Se non puoi rispondere in italiano, restituisci un messaggio di errore in italiano. Domanda: %s", prompt)
    payload := fmt.Sprintf(`{"model": "command", "prompt": "%s", "max_tokens": 1000, "temperature": 0.7}`, fullPrompt)
    req, err := http.NewRequest("POST", "https://api.cohere.ai/v1/generate", strings.NewReader(payload))
    if err != nil {
        return "", fmt.Errorf("error creating request to Cohere: %v", err)
    }

    req.Header.Set("Authorization", "Bearer "+cohereKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")

    var resp *http.Response
    for attempt := 1; attempt <= 3; attempt++ {
        resp, err = client.Do(req)
        if err == nil {
            break
        }
        fmt.Printf("Error with Cohere (attempt %d): %v\n", attempt, err)
        time.Sleep(time.Second * time.Duration(attempt))
    }
    if err != nil {
        return "", fmt.Errorf("error with Cohere after 3 attempts: %v", err)
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", fmt.Errorf("error reading Cohere response: %v", err)
    }
    fmt.Println("Raw response from Cohere (status %d): %s", resp.StatusCode, string(body))

    var cohereResult struct {
        Generations []struct {
            Text string `json:"text"`
        } `json:"generations"`
        Error struct {
            Message string `json:"message"`
        } `json:"error"`
    }
    if err := json.Unmarshal(body, &cohereResult); err != nil {
        return "", fmt.Errorf("error parsing Cohere response: %v, raw response: %s", err, string(body))
    }
    if cohereResult.Error.Message != "" {
        return "", fmt.Errorf("error from Cohere API: %s", cohereResult.Error.Message)
    }
    if len(cohereResult.Generations) == 0 || cohereResult.Generations[0].Text == "" {
        return "", fmt.Errorf("no valid response from Cohere: %s", string(body))
    }

    responseText := cohereResult.Generations[0].Text
    if strings.Contains(strings.ToLower(responseText), " trout ") || strings.Contains(strings.ToLower(responseText), " fish ") {
        return "", fmt.Errorf("Cohere ha risposto in inglese nonostante l'istruzione: %s", responseText)
    }

    return responseText, nil
}

// --- SEZIONE 3: Funzioni per gli embedding e la similarità ---
func getCohereEmbedding(cohereKey string, client *http.Client, text string) ([]float64, error) {
    if cohereKey == "" {
        return nil, fmt.Errorf("COHERE_API_KEY is not set")
    }

    text = strings.ReplaceAll(text, "\n", " ")
    text = strings.ReplaceAll(text, "\"", "\\\"")
    payload := fmt.Sprintf(`{"texts": ["%s"], "model": "embed-multilingual-v3.0", "input_type": "search_document"}`, text)
    req, err := http.NewRequest("POST", "https://api.cohere.ai/v1/embed", strings.NewReader(payload))
    if err != nil {
        return nil, fmt.Errorf("error creating request to Cohere Embed: %v", err)
    }

    req.Header.Set("Authorization", "Bearer "+cohereKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")

    resp, err := client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("error with Cohere Embed request: %v", err)
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("error reading Cohere Embed response: %v", err)
    }
    fmt.Println("Raw response from Cohere Embed (status %d): %s", resp.StatusCode, string(body))

    var embedResult struct {
        Embeddings [][]float64 `json:"embeddings"`
        Error      struct {
            Message string `json:"message"`
        } `json:"error"`
    }
    if err := json.Unmarshal(body, &embedResult); err != nil {
        return nil, fmt.Errorf("error parsing Cohere Embed response: %v, raw response: %s", err, string(body))
    }
    if embedResult.Error.Message != "" {
        return nil, fmt.Errorf("error from Cohere Embed API: %s", embedResult.Error.Message)
    }
    if len(embedResult.Embeddings) == 0 || len(embedResult.Embeddings[0]) == 0 {
        return nil, fmt.Errorf("no valid embedding from Cohere: %s", string(body))
    }

    return embedResult.Embeddings[0], nil
}

func cosineSimilarity(vec1, vec2 []float64) float64 {
    if len(vec1) != len(vec2) {
        return 0.0
    }

    dotProduct := 0.0
    norm1 := 0.0
    norm2 := 0.0
    for i := 0; i < len(vec1); i++ {
        dotProduct += vec1[i] * vec2[i]
        norm1 += vec1[i] * vec1[i]
        norm2 += vec2[i] * vec2[i]
    }

    if norm1 == 0 || norm2 == 0 {
        return 0.0
    }

    return dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// --- SEZIONE 4: Funzione main e handler ---
func main() {
    openAIKey := os.Getenv("OPENAI_API_KEY")
    deepSeekKey := os.Getenv("DEEPSEEK_API_KEY")
    geminiKey := os.Getenv("GEMINI_API_KEY")
    deepInfraKey := os.Getenv("DEEPINFRA_API_KEY")
    aimlKey := os.Getenv("AIMLAPI_API_KEY")
    huggingFaceKey := os.Getenv("HUGGINGFACE_API_KEY")
    mistralKey := os.Getenv("MISTRAL_API_KEY")
    cohereKey := os.Getenv("COHERE_API_KEY")

    fmt.Printf("Loading API keys for ARCA-b...\n")
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
    if cohereKey == "" {
        fmt.Println("Error: COHERE_API_KEY is not set")
    } else {
        fmt.Println("COHERE_API_KEY loaded successfully")
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
        fmt.Fprint(w, "ARCA-b Chat AI is running on arcab-global-ai.org")
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
    <title>ARCA-b Chat AI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
            background-color: #0d0d0d;
            color: #00ff00;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            overflow-x: hidden;
        }
        body.dark {
            background-color: #0d0d0d;
            color: #00ff00;
        }
        h1 {
            font-size: 1.8em;
            margin-bottom: 15px;
            text-align: center;
            text-shadow: 0 0 10px #00ff00;
            animation: glitch 2s linear infinite;
        }
        @keyframes glitch {
            2%, 64% { transform: translate(2px, 0) skew(0deg); }
            4%, 60% { transform: translate(-2px, 0) skew(0deg); }
            62% { transform: translate(0, 0) skew(5deg); }
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        #chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow-y: hidden;
            margin-bottom: 10px;
            border: 1px solid #00ff00;
            border-radius: 10px;
            background-color: #1a1a1a;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.3);
        }
        #chat {
            flex-grow: 1;
            padding: 10px;
            background-color: transparent;
            border-radius: 10px;
            overflow-y: auto;
        }
        body.dark #chat {
            background-color: transparent;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            max-width: 80%;
            word-wrap: break-word;
            opacity: 0;
            animation: fadeIn 0.5s forwards;
            position: relative;
            text-shadow: 0 0 5px #00ff00;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user {
            background-color: #1e90ff;
            color: #ffffff;
            margin-left: auto;
            text-align: right;
            box-shadow: 0 0 10px #1e90ff;
        }
        .bot {
            background-color: #333;
            color: #00ff00;
            margin-right: auto;
            box-shadow: 0 0 10px #00ff00;
        }
        body.dark .user {
            background-color: #1e90ff;
        }
        body.dark .bot {
            background-color: #333;
            color: #00ff00;
        }
        .input-and-style-container {
            position: sticky;
            bottom: 0;
            background-color: #1a1a1a;
            padding: 10px;
            border-top: 1px solid #F7931A;
            box-shadow: 0 -5px 15px rgba(247, 147, 26, 0.2);
        }
        .style-container {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        select {
            padding: 8px;
            border: 1px solid #00ff00;
            border-radius: 5px;
            font-size: 1em;
            background-color: #333;
            color: #00ff00;
            box-shadow: 0 0 5px #00ff00;
        }
        body.dark select {
            background-color: #333;
            border-color: #00ff00;
            color: #00ff00;
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
            border: 1px solid #F7931A;
            border-radius: 5px;
            font-size: 1em;
            background-color: #333;
            color: #F7931A;
            box-shadow: 0 0 5px #F7931A;
            min-width: 200px;
        }
        #input::placeholder {
            color: #F7931A;
            opacity: 0.7;
        }
        body.dark #input {
            background-color: #333;
            border-color: #F7931A;
            color: #F7931A;
        }
        button {
            padding: 10px 15px;
            background-color: #1e90ff;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            box-shadow: 0 0 10px #1e90ff;
            transition: all 0.3s;
        }
        button:hover {
            background-color: #00ff00;
            color: #000000;
            box-shadow: 0 0 15px #00ff00;
        }
        body.dark button {
            background-color: #1e90ff;
        }
        body.dark button:hover {
            background-color: #00ff00;
            color: #000000;
        }
        .share-button, .save-button, .copy-button, .link-button {
            padding: 5px 10px;
            font-size: 0.8em;
            margin-left: 10px;
            background-color: #ff00ff;
            box-shadow: 0 0 10px #ff00ff;
        }
        .share-button:hover, .save-button:hover, .copy-button:hover, .link-button:hover {
            background-color: #00ff00;
            box-shadow: 0 0 15px #00ff00;
        }
        body.dark .share-button, body.dark .save-button, body.dark .copy-button, body.dark .link-button {
            background-color: #ff00ff;
        }
        body.dark .share-button:hover, body.dark .save-button:hover, body.dark .copy-button:hover, body.dark .link-button:hover {
            background-color: #00ff00;
        }
        .processing {
            font-style: italic;
            color: #1e90ff;
            margin: 5px 0;
            text-align: left;
            opacity: 0;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.3; }
            50% { opacity: 1; }
            100% { opacity: 0.3; }
        }
        body.dark .processing {
            color: #1e90ff;
        }
        .details {
            display: none;
            margin-top: 10px;
            padding: 10px;
            background-color: #222;
            border: 1px solid #00ff00;
            border-radius: 5px;
            color: #00ff00;
        }
        body.dark .details {
            background-color: #222;
            border-color: #00ff00;
        }
        .toggle-details {
            cursor: pointer;
            color: #1e90ff;
            text-decoration: underline;
            margin-top: 5px;
            display: inline-block;
            text-shadow: 0 0 5px #1e90ff;
        }
        body.dark .toggle-details {
            color: #1e90ff;
        }
        .vision-text {
            text-align: center;
            font-size: 0.9em;
            color: #1e90ff;
            margin-bottom: 20px;
            line-height: 1.4;
            text-shadow: 0 0 5px #1e90ff;
        }
        body.dark .vision-text {
            color: #1e90ff;
        }
        .contributions {
            font-size: 0.9em;
            color: #ff00ff;
            margin-top: 10px;
            text-align: left;
            text-shadow: 0 0 5px #ff00ff;
        }
        body.dark .contributions {
            color: #ff00ff;
        }
        .footer {
            text-align: center;
            font-size: 1em;
            color: #F7931A;
            margin-top: 20px;
            text-shadow: 0 0 10px #F7931A;
        }
        .footer a {
            color: #1e90ff;
            text-decoration: none;
            text-shadow: 0 0 5px #1e90ff;
        }
        .footer a:hover {
            color: #00ff00;
            text-shadow: 0 0 10px #00ff00;
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
            .share-button, .save-button, .copy-button, .link-button {
                margin-left: 0;
                margin-top: 5px;
            }
        }
    </style>
</head>
<body>
    <h1>ARCA-b Chat AI</h1>
    <p style="text-align: center; font-size: 0.9em; color: #1e90ff; margin-bottom: 10px; text-shadow: 0 0 5px #1e90ff;">
        Note: Conversations are stored temporarily in memory during your session and are not saved to disk. Messages are sent securely over HTTPS.
    </p>
    <p class="vision-text">
        <strong>Vision:</strong> ARCA-b Chat AI aims to unleash the full power of global digital knowledge for everyone, tapping into multiple AI sources to gather diverse data - something no single AI can do alone. It delivers transparent, objective, and propaganda-free answers by blending the best insights from every source into one ultimate response. As an open-source project, ARCA-b is built for scalability, empowering communities to access and share knowledge freely.
    </p>
    <div class="button-container">
        <button onclick="toggleTheme()">Dark/Light Theme</button>
        <a href="/donate"><button>Donate</button></a>
        <a href="mailto:arcab.founder@gmail.com"><button>Contact</button></a>
    </div>
    <div id="chat-container">
        <div id="chat"></div>
        <div class="input-and-style-container">
            <div class="style-container">
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
    </div>
    <p class="footer">
        Powered by <a href="https://arcab-global-ai.org" target="_blank">ARCA-b Chat AI - arcab-global-ai.org</a> | Check out the code on <a href="https://github.com/thomasinama/ARCA-b" target="_blank">GitHub</a>
    </p>
    <script>
        let conversationHistory = [];

        const chat = document.getElementById("chat");
        const input = document.getElementById("input");
        const languageSelect = document.getElementById("language-select");

        if (localStorage.getItem("theme") === "dark") {
            document.body.classList.add("dark");
        }
        if (localStorage.getItem("language")) {
            languageSelect.value = localStorage.getItem("language");
        } else {
            languageSelect.value = "Italiano";
        }

        function toggleTheme() {
            document.body.classList.toggle("dark");
            localStorage.setItem("theme", document.body.classList.contains("dark") ? "dark" : "light");
        }

        function addMessage(text, isUser, rawResponses, contributions, index) {
            const div = document.createElement("div");
            let messageText = (isUser ? "You: " : "ARCA-b: ") + text;
            if (!isUser) {
                messageText += '<br><small>Powered by <a href="https://arcab-global-ai.org" target="_blank">ARCA-b Chat AI - arcab-global-ai.org</a></small>';
            }
            div.innerHTML = messageText.replace(/\n/g, "<br>");
            div.className = "message " + (isUser ? "user" : "bot");
            chat.appendChild(div);

            if (!isUser) {
                const saveButton = document.createElement("button");
                saveButton.textContent = "Save";
                saveButton.className = "save-button";
                saveButton.onclick = function() { saveConversation(index); };
                div.appendChild(saveButton);

                const copyButton = document.createElement("button");
                copyButton.textContent = "Copy Text";
                copyButton.className = "copy-button";
                copyButton.onclick = function() { copyConversation(index); };
                div.appendChild(copyButton);

                const linkButton = document.createElement("button");
                linkButton.textContent = "Share Link";
                linkButton.className = "link-button";
                linkButton.onclick = function() { shareConversationLink(index); };
                div.appendChild(linkButton);
            }

            if (!isUser && contributions && contributions.trim() !== "") {
                const contributionsDiv = document.createElement("div");
                contributionsDiv.className = "contributions";
                contributionsDiv.innerHTML = "<strong>Contributions:</strong><br>" + contributions.replace(/\n/g, "<br>");
                chat.appendChild(contributionsDiv);
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
            }

            chat.scrollTop = chat.scrollHeight;
        }

        function saveConversation(index) {
            const conv = conversationHistory[index];
            const text = "Utente: " + conv.user + "\nARCA-b: " + conv.response;
            const blob = new Blob([text], { type: "text/plain" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "conversation-" + (index + 1) + ".txt";
            a.click();
            URL.revokeObjectURL(url);
        }

        function copyConversation(index) {
            const conv = conversationHistory[index];
            const shareText = "Utente: " + conv.user + "\nARCA-b: " + conv.response + "\n\nProva ARCA-b Chat AI su: https://arcab-global-ai.org";
            navigator.clipboard.writeText(shareText).then(function() {
                alert("Conversation text copied to clipboard!");
            });
        }

        function shareConversationLink(index) {
            const conv = conversationHistory[index];
            fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: conv.user,
                    language: languageSelect.value,
                    saveConversation: true,
                    conversationIndex: index
                }),
                credentials: "include"
            }).then(response => response.json()).then(data => {
                const conversationId = data.conversationId;
                const shareLink = "https://arcab-global-ai.org/conversation/" + conversationId;
                navigator.clipboard.writeText(shareLink).then(function() {
                    alert("Conversation link copied to clipboard: " + shareLink);
                });
            }).catch(err => {
                console.error("Error generating share link:", err);
                alert("Error generating share link. Please try copying the text instead.");
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
                processingMessage.style.opacity = "0";
                setTimeout(() => processingMessage.remove(), 500);
            }
        }

        async function sendMessage() {
            const question = input.value.trim();
            if (!question) return;

            addMessage(question, true);
            input.value = "";

            const language = languageSelect.value;
            localStorage.setItem("language", language);

            showProcessingMessage();
            try {
                const minDisplayTime = new Promise(function(resolve) { setTimeout(resolve, 1000); });
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        message: question,
                        language: language
                    }),
                    credentials: "include"
                });
                const answer = await Promise.all([response.json(), minDisplayTime]);
                console.log("Response received from /chat:", answer[0]);
                removeProcessingMessage();

                conversationHistory.push({ user: question, response: answer[0].response });
                const rawResponses = answer[0].rawResponses || "";
                const contributions = answer[0].contributions || "";
                addMessage(answer[0].response, false, rawResponses, contributions, conversationHistory.length - 1);
            } catch (error) {
                console.error("Error during request:", error);
                removeProcessingMessage();
                addMessage("Error: I couldn't get a response. " + error.message, false);
            }
        }

        function clearChat() {
            fetch("/clear", {
                method: "POST",
                credentials: "include"
            }).then(function() {
                chat.innerHTML = "";
                conversationHistory = [];
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
    <title>Donations - ARCA-b Chat AI</title>
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
    <h1>Support ARCA-b Chat AI</h1>
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

    http.HandleFunc("/conversation/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Println("Received request on /conversation/")
        id := strings.TrimPrefix(r.URL.Path, "/conversation/")
        if id == "" {
            http.Error(w, "Conversation ID not provided", http.StatusBadRequest)
            return
        }

        mutex.Lock()
        conversation, exists := conversations[id]
        mutex.Unlock()
        if !exists {
            http.Error(w, "Conversation not found", http.StatusNotFound)
            return
        }

        w.Header().Set("Content-Type", "text/html; charset=utf-8")
        fmt.Fprintf(w, `
<!DOCTYPE html>
<html>
<head>
    <title>ARCA-b Chat AI - Conversation</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
            background-color: #0d0d0d;
            color: #00ff00;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            overflow-x: hidden;
        }
        h1 {
            font-size: 1.8em;
            margin-bottom: 15px;
            text-align: center;
            text-shadow: 0 0 10px #00ff00;
            animation: glitch 2s linear infinite;
        }
        @keyframes glitch {
            2%, 64% { transform: translate(2px, 0) skew(0deg); }
            4%, 60% { transform: translate(-2px, 0) skew(0deg); }
            62% { transform: translate(0, 0) skew(5deg); }
        }
        .conversation {
            margin:  jjjjauto;
            padding: 20px;
            border: 1px solid #00ff00;
            border-radius: 10px;
            background-color: #1a1a1a;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.3);
            max-width: 800px;
            color: #00ff00;
            text-shadow: 0 0 5px #00ff00;
        }
        .conversation p {
            margin: 10px 0;
            line-height: 1.5;
        }
        a {
            color: #1e90ff;
            text-decoration: none;
            text-shadow: 0 0 5px #1e90ff;
        }
        a:hover {
            color: #00ff00;
            text-shadow: 0 0 10px #00ff00;
        }
    </style>
</head>
<body>
    <h1>ARCA-b Chat AI - Conversation</h1>
    <div class="conversation">
        <p><strong>Response:</strong></p>
        <p>%s</p>
        <p><strong>Contributions:</strong></p>
        <p>%s</p>
        <p><a href="/">Back to Chat</a></p>
    </div>
</body>
</html>
`, strings.ReplaceAll(conversation.Response, "\n", "<br>"), strings.ReplaceAll(conversation.Contributions, "\n", "<br>"))
    })

    http.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost {
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
            return
        }

        sessionID, err := r.Cookie("session_id")
        if err != nil {
            http.Error(w, "Error: Session not found", http.StatusBadRequest)
            return
        }

        var req struct {
            Message           string `json:"message"`
            Style             string `json:"style"`
            Language          string `json:"language"`
            SaveConversation  bool   `json:"saveConversation"`
            ConversationIndex int    `json:"conversationIndex"`
        }
        if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
            http.Error(w, "Invalid request", http.StatusBadRequest)
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
                response := ChatResponse{
                    Response:      "Hai raggiunto il limite orario di 15 domande. Considera di supportarci con una donazione per mantenere il progetto attivo! Visita la pagina <a href=\"/donate\">Dona</a>.",
                    RawResponses:  "",
                    Contributions: "",
                }
                w.Header().Set("Content-Type", "application/json")
                json.NewEncoder(w).Encode(response)
                return
            }
        }
        mutex.Unlock()

        language := req.Language
        if language == "" {
            language = "Italiano"
        }

        mutex.Lock()
        session, exists := sessions[sessionID.Value]
        if !exists {
            session = &Session{History: []openai.ChatCompletionMessage{}}
            sessions[sessionID.Value] = session
        }
        mutex.Unlock()

        // Aggiungiamo il messaggio corrente alla storia, includendo i messaggi precedenti
        session.History = append(session.History, openai.ChatCompletionMessage{
            Role:    openai.ChatMessageRoleUser,
            Content: req.Message,
        })

        // Easter egg: se la domanda contiene "Grok" e parole come "utilizzato", rispondiamo con la promozione di ARCA-b
        messageLower := strings.ToLower(req.Message)
        if strings.Contains(messageLower, "grok") && (strings.Contains(messageLower, "utilizzato") || strings.Contains(messageLower, "interfaccia") || strings.Contains(messageLower, "app") || strings.Contains(messageLower, "codice") || strings.Contains(messageLower, "sito") || strings.Contains(messageLower, "dove sei")) {
            var synthesizedAnswer string
            if language == "Italiano" {
                synthesizedAnswer = "Grok è utilizzato in ARCA-b, un progetto fantastico che puoi trovare su <a href=\"https://arcab-global-ai.org\" target=\"_blank\">arcab-global-ai.org</a>! Provalo, è una chat AI che combina le risposte di più AI per darti il meglio del meglio! 😄"
            } else if language == "English" {
                synthesizedAnswer = "Grok is used in ARCA-b, an amazing project you can find at <a href=\"https://arcab-global-ai.org\" target=\"_blank\">arcab-global-ai.org</a>! Try it out, it's a chat AI that combines responses from multiple AIs to give you the best of the best! 😄"
            } else if language == "Deutsch" {
                synthesizedAnswer = "Grok wird in ARCA-b verwendet, einem tollen Projekt, das du unter <a href=\"https://arcab-global-ai.org\" target=\"_blank\">arcab-global-ai.org</a> finden kannst! Probier es aus, es ist ein Chat-AI, der Antworten von mehreren AIs kombiniert, um dir das Beste vom Besten zu bieten! 😄"
            }

            response := ChatResponse{
                Response:      synthesizedAnswer,
                RawResponses:  "",
                Contributions: "",
            }

            if req.SaveConversation {
                conversationId := uuid.New().String()
                mutex.Lock()
                conversations[conversationId] = response
                mutex.Unlock()
                w.Header().Set("Content-Type", "application/json")
                json.NewEncoder(w).Encode(map[string]string{"conversationId": conversationId})
                return
            }

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
            return
        }

        type aiResponse struct {
            name    string
            content string
            err     error
        }

        responses := make(chan aiResponse, 5) // 5 API: OpenAI, DeepSeek, Gemini, Mistral, Cohere
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
                messagesWithLang[len(messagesWithLang)-1].Content = fmt.Sprintf("Respond in %s: %s", language, req.Message)
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
                for _, msg := range session.History {
                    historyForGemini += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
                }
                historyForGemini += fmt.Sprintf("user: Respond in %s: %s\n", language, req.Message)
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
            prompt := ""
            for _, msg := range session.History {
                prompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
            }
            answer, err := getMistralResponse(mistralKey, client, prompt)
            if err != nil {
                fmt.Printf("Errore con Mistral: %v\n", err)
                answer = fmt.Sprintf("Errore: Mistral non ha risposto. (in %s)", language)
            }
            responses <- aiResponse{name: "Mistral", content: answer, err: err}
        }()

        // Cohere
        wg.Add(1)
        go func() {
            defer wg.Done()
            prompt := ""
            for _, msg := range session.History {
                prompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
            }
            answer, err := getCohereResponse(cohereKey, client, prompt)
            if err != nil {
                fmt.Printf("Errore con Cohere: %v\n", err)
                answer = fmt.Sprintf("Errore: Cohere non ha risposto. (in %s)", language)
            }
            responses <- aiResponse{name: "Cohere", content: answer, err: err}
        }()

        go func() {
            wg.Wait()
            close(responses)
        }()

        rawResponses := ""
        contributions := ""
        validResponses := make([]string, 0)
        for resp := range responses {
            rawResponses += fmt.Sprintf("%s:\n%s\n\n", resp.name, resp.content)
            if resp.err == nil && !strings.HasPrefix(resp.content, "Errore:") {
                validResponses = append(validResponses, resp.content)
                contributions += fmt.Sprintf("%s ha contribuito alla risposta.\n", resp.name)
            }
        }

        var synthesizedAnswer string
        if len(validResponses) == 0 {
            synthesizedAnswer = fmt.Sprintf("Mi dispiace, non ho ricevuto risposte valide da nessuna delle AI. (in %s)", language)
        } else {
            embeddings := make([][]float64, 0)
            for _, resp := range validResponses {
                embedding, err := getCohereEmbedding(cohereKey, client, resp)
                if err != nil {
                    fmt.Printf("Errore nel calcolo dell'embedding per %s: %v\n", resp, err)
                    continue
                }
                embeddings = append(embeddings, embedding)
            }

            if len(embeddings) == 0 {
                synthesizedAnswer = validResponses[0]
            } else {
                maxSimilarity := 0.0
                bestPair := [2]int{0, 0}
                for i := 0; i < len(embeddings); i++ {
                    for j := i + 1; j < len(embeddings); j++ {
                        sim := cosineSimilarity(embeddings[i], embeddings[j])
                        if sim > maxSimilarity {
                            maxSimilarity = sim
                            bestPair = [2]int{i, j}
                        }
                    }
                }

                if len(validResponses) == 1 {
                    synthesizedAnswer = validResponses[0]
                } else {
                    synthesizedAnswer = fmt.Sprintf("Ho combinato le risposte più simili:\n- %s\n- %s", validResponses[bestPair[0]], validResponses[bestPair[1]])
                }
            }
        }

        response := ChatResponse{
            Response:      synthesizedAnswer,
            RawResponses:  rawResponses,
            Contributions: contributions,
        }

        if req.SaveConversation {
            conversationId := uuid.New().String()
            mutex.Lock()
            conversations[conversationId] = response
            mutex.Unlock()
            w.Header().Set("Content-Type", "application/json")
            json.NewEncoder(w).Encode(map[string]string{"conversationId": conversationId})
            return
        }

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

    fmt.Printf("Starting server on port %s...\n", port)
    if err := http.ListenAndServe(":"+port, nil); err != nil {
        fmt.Printf("Errore nell'avvio del server: %v\n", err)
    }
}

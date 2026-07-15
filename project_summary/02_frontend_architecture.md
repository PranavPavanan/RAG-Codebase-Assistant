# Frontend Architecture

The frontend is a Single Page Application (SPA) built entirely with Vanilla JavaScript (`frontend/app.js`), eschewing heavy frameworks like React or Vue. 

## State Management and Core Logic
The main application logic is encapsulated inside a monolithic `RAGApp` class. It manages state via a centralized `this.state` object:

```javascript
this.state = {
    selectedRepository: null,
    isIndexing: false,
    messages: [],
    conversationId: null,
    sessionId: null,
    currentConversation: null,
    conversations: [],
    // ...
}
```

The app handles view routing manually by toggling `display: none` on specific panel containers (`#panel-search`, `#panel-index`, `#panel-chat`).

## Nuance: In-Memory Multithreaded Chat
While seemingly simple, the frontend implements a sophisticated **in-memory multithreaded chat system**. This is a critical nuance for any engineer working on this codebase:

1. **Conversation Isolation:** 
   The frontend UI supports multiple chat tabs. These are tracked in the `this.state.conversations` array. Each object in this array represents a distinct thread with its own `id` and `messages` array.
   
2. **Conversation IDs:**
   When a user clicks "New Chat", the frontend generates a unique pseudo-ID (e.g., `conv-1698765432100`) and pushes a new conversation object to the array. It updates `this.state.conversationId` to this new ID.

3. **Backend Communication:**
   During a query submission, `app.js` calls `ApiClient.chatQuery(query, this.state.conversationId, this.state.sessionId)`. By explicitly sending the frontend-generated `conversation_id`, it tells the backend exactly which conversational context string to append the new message to.

4. **Context Switching:**
   Clicking a different conversation tab triggers `switchToConversation(id)`. This swaps `this.state.messages` to the array of the selected thread and re-renders the DOM, creating a seamless multi-thread experience without requiring backend persistence.

## API Integration
The `ApiClient` class wraps the native `fetch` API. It handles JSON serialization, error extraction, and routes directly to the FastAPI endpoints (`http://localhost:8000/api/...`).

## WebSockets
For real-time indexing feedback, the frontend establishes a WebSocket connection to `ws://localhost:8000/api/index/ws/{taskId}`. It receives JSON payloads tracking files processed, total files, and percentage, which are then mapped directly to DOM elements (progress bars, status text) to provide a fluid user experience.

/**
 * API client untuk backend Chatbot RAG (FastAPI).
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// --- Types ---

export interface IngestResponse {
  session_id: string;
  document_id: string;
  file_name: string;
  pages_loaded: number;
  chunks_created: number;
  points_upserted: number;
  embed_dim: number;
  chunk_size: number;
  chunk_overlap: number;
  duration_seconds: number;
}

export interface SourceCitation {
  file_name: string | null;
  page_label: string | null;
  score: number;
}

export type ChatEvent =
  | { type: "sources"; sources: SourceCitation[] }
  | { type: "delta"; text: string }
  | { type: "stage"; stage: "retrieval" | "generation"; status: "start" | "done"; duration_seconds?: number; chunks_found?: number }
  | { type: "done"; used_fallback: boolean; duration_seconds: number }
  | { type: "error"; detail: string };

export interface StatsResponse {
  document_count: number;
  chunk_count: number;
  session_count: number;
  message_count: number;
  fallback_rate: number;
}

export async function getStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_BASE_URL}/stats`);
  if (!response.ok) throw new Error(`Gagal memuat statistik (HTTP ${response.status})`);
  return response.json() as Promise<StatsResponse>;
}

export interface SessionSummary {
  id: string;
  title: string;
  created_at: string;
}

export interface MessageRecord {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: SourceCitation[];
  used_fallback: boolean;
  created_at: string;
}

// --- API calls ---

export async function uploadDocument(file: File): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/documents`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({ detail: "Terjadi kesalahan tidak diketahui" }));
    throw new Error(errorBody.detail ?? `HTTP ${response.status}`);
  }

  return response.json() as Promise<IngestResponse>;
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await fetch(`${API_BASE_URL}/sessions`);
  if (!response.ok) throw new Error(`Gagal memuat riwayat chat (HTTP ${response.status})`);
  return response.json() as Promise<SessionSummary[]>;
}

export async function getSessionMessages(sessionId: string): Promise<MessageRecord[]> {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/messages`);
  if (!response.ok) throw new Error(`Gagal memuat riwayat pesan (HTTP ${response.status})`);
  return response.json() as Promise<MessageRecord[]>;
}

/**
 * Kirim pertanyaan dan terima jawaban secara streaming (SSE via fetch --
 * EventSource native tidak support POST).
 */
export async function streamChat(
  sessionId: string,
  question: string,
  topK: number | undefined,
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question, top_k: topK }),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({ detail: "Terjadi kesalahan tidak diketahui" }));
    throw new Error(errorBody.detail ?? `HTTP ${response.status}`);
  }

  if (!response.body) {
    throw new Error("Response tidak punya body -- browser mungkin tidak mendukung streaming.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const processRawEvent = (rawEvent: string) => {
    const line = rawEvent.trim();
    if (!line.startsWith("data: ")) return;
 
    const jsonStr = line.slice("data: ".length);
    try {
      const parsedEvent = JSON.parse(jsonStr) as ChatEvent;
      onEvent(parsedEvent);
    } catch (err) {
      console.error("Gagal parse SSE event:", jsonStr, err);
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const rawEvents = buffer.split("\n\n");
    buffer = rawEvents.pop() ?? "";

    for (const rawEvent of rawEvents) {
      processRawEvent(rawEvent)
    }
    // Proses sisa buffer setelah stream benar-benar selesai
    if (buffer.trim()) {
      processRawEvent(buffer);
  }
  }
}
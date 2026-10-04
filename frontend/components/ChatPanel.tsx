"use client";

import { useCallback, useEffect, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import { getSessionMessages, streamChat, type ChatEvent, type SourceCitation } from "@/lib/api";
import Form from "@/components/Form";
import { Bot, FileText } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

interface ChatTurn {
  question: string;
  answer: string;
  sources: SourceCitation[];
  usedFallback: boolean;
  isGenerating: boolean;
  isRevealing: boolean;
  error: string | null;
}

interface ChatPanelProps {
  sessionId: string | null;
  disabled?: boolean;
}

const REVEAL_INTERVAL_MS = 35;
const SUGGESTED_QUESTIONS = [
  "Ringkas isi dokumen ini",
  "Apa poin-poin pentingnya?",
  "Jelaskan topik utama dokumen",
];

/** Animasikan `fullText` (string yang SUDAH LENGKAP & pasti benar) secara
 * bertahap per kata. Karena sumbernya sudah pasti utuh (bukan potongan
 * network yang mungkin hilang/salah urai), fungsi ini jauh lebih sederhana
 * & tidak rentan bug dibanding versi reveal-queue sebelumnya. */
function animateReveal(fullText: string, onUpdate: (partial: string) => void,): Promise<void> {
  return new Promise((resolve) => {
    let position = 0;
    const id = setInterval(() => {
      if (position >= fullText.length) {
        clearInterval(id);
        resolve();
        return;
      }
      const spaceIndex = fullText.indexOf(" ", position);
      position = spaceIndex === -1 ? fullText.length : spaceIndex + 1;
      onUpdate(fullText.slice(0, position));
    }, REVEAL_INTERVAL_MS);
  });
}

export default function ChatPanel({ sessionId, disabled = false }: ChatPanelProps) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(Boolean(sessionId));
  const turnsLengthRef = useRef(0);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const followOutputRef = useRef(true);

  useEffect(() => {
    turnsLengthRef.current = turns.length;
  }, [turns.length]);

  const scrollToBottom = useCallback(() => {
    if (!followOutputRef.current) return;
    const messagesContainer = messagesContainerRef.current;
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  }, []);

  useEffect(() => {
    window.addEventListener("resize", scrollToBottom);
    return () => window.removeEventListener("resize", scrollToBottom);
  }, [scrollToBottom]);

  useEffect(() => {
    let secondFrame = 0;
    const frame = requestAnimationFrame(() => {
      scrollToBottom();
      secondFrame = requestAnimationFrame(() => {
        scrollToBottom();
      });
    });

    return () => {
      cancelAnimationFrame(frame);
      cancelAnimationFrame(secondFrame);
    };
  }, [turns, isLoadingHistory, scrollToBottom]);

  useEffect(() => {
    if (!sessionId) return;

    let cancelled = false;

    getSessionMessages(sessionId)
      .then((messages) => {
        if (cancelled) return;

        const loadedTurns: ChatTurn[] = [];
        for (let i = 0; i < messages.length; i++) {
          if (messages[i].role !== "user") continue;
          const userMsg = messages[i];
          const assistantMsg = messages[i + 1]?.role === "assistant" ? messages[i + 1] : null;

          loadedTurns.push({
            question: userMsg.content,
            answer: assistantMsg?.content ?? "",
            sources: assistantMsg?.sources ?? [],
            usedFallback: assistantMsg?.used_fallback ?? false,
            isGenerating: false,
            isRevealing: false,
            error: null,
          });
        }
        setTurns(loadedTurns);
      })
      .catch((err) => console.error("Gagal memuat riwayat pesan:", err))
      .finally(() => {
        if (!cancelled) setIsLoadingHistory(false);
      });

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const askQuestion = useCallback(async (suggestedQuestion?: string) => {
    const question = (suggestedQuestion ?? input).trim();
    if (!question || isBusy || disabled || !sessionId) return;
 
    setInput("");
    setIsBusy(true);
 
    const turnIndex = turnsLengthRef.current;
    setTurns((prev) => [
      ...prev,
      {
        question,
        answer: "",
        sources: [],
        usedFallback: false,
        isGenerating: true,
        isRevealing: false,
        error: null,
      },
    ]);
 
    const patchTurn = (patch: Partial<ChatTurn>) => {
      setTurns((prev) => {
        const next = [...prev];
        next[turnIndex] = { ...next[turnIndex], ...patch };
        return next;
      });
    };
 
    try {
      let streamError: string | null = null;
 
      // Konsumsi stream HANYA untuk: (a) tampilkan sources sedini mungkin,
      // (b) deteksi error di tengah generation. Event "delta" sengaja
      // diabaikan -- lihat catatan desain di atas.
      await streamChat(sessionId, question, undefined, (event: ChatEvent) => {
        if (event.type === "sources") {
          patchTurn({ sources: event.sources });
        } else if (event.type === "error") {
          streamError = event.detail;
        }
      });
 
      if (streamError) {
        patchTurn({ error: streamError, isGenerating: false });
        return;
      }
 
      // Stream selesai TANPA error -> backend sudah simpan pesan assistant
      // lengkap ke database. Ambil dari sana (sumber kebenaran), bukan dari
      // rakitan delta frontend.
      const messages = await getSessionMessages(sessionId);
      const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
 
      patchTurn({ isGenerating: false, isRevealing: true });
 
      if (lastAssistant) {
        await animateReveal(lastAssistant.content, (partial) => patchTurn({ answer: partial }));
        patchTurn({
          sources: lastAssistant.sources,
          usedFallback: lastAssistant.used_fallback,
          isRevealing: false,
        });
      } else {
        // Kondisi tak terduga: stream sukses tapi pesan assistant tidak
        // ketemu di DB -- tampilkan sebagai error, jangan diam-diam kosong.
        patchTurn({
          isRevealing: false,
          error: "Jawaban selesai dibuat tapi gagal dimuat ulang. Coba refresh halaman.",
        });
      }
    } catch (err) {
      patchTurn({
        error: err instanceof Error ? err.message : "Gagal menghubungi server.",
        isGenerating: false,
        isRevealing: false,
      });
    } finally {
      setIsBusy(false);
    }
  }, [input, isBusy, disabled, sessionId]);
 
  const onSubmit = useCallback(
    (e: FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      askQuestion();
    },
    [askQuestion],
  );

  const onInputChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  }, []);
  
  return (
    <section className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-col">
      <div
        ref={messagesContainerRef}
        onScroll={(event) => {
          const container = event.currentTarget;
          followOutputRef.current =
            container.scrollHeight - container.scrollTop - container.clientHeight < 80;
        }}
        className={`chat-scrollbar min-h-0 flex-1 overflow-y-auto overscroll-contain px-1 ${
          turns.length === 0 && !isLoadingHistory
            ? "flex flex-col items-center justify-center"
            : "py-6 sm:py-8"
        }`}
        aria-live="polite"
        aria-busy={isLoadingHistory || isBusy}
      >
        {isLoadingHistory && (
          <div className="mx-auto flex w-full max-w-2xl flex-col gap-8 py-8" role="status">
            <span className="sr-only">Memuat riwayat percakapan...</span>
            {[0, 1, 2].map((item) => (
              <div key={item} className="animate-pulse space-y-3">
                <div className="ml-auto h-10 w-2/3 rounded-2xl bg-(--color-muted)" />
                <div className="h-4 w-full max-w-xl rounded bg-(--color-muted)" />
                <div className="h-4 w-4/5 max-w-lg rounded bg-(--color-muted)" />
              </div>
            ))}
          </div>
        )}

        {!isLoadingHistory && turns.length === 0 && (
          <div className="w-full max-w-xl px-2 py-10 text-center">
            <div className="mx-auto mb-5 flex size-12 items-center justify-center rounded-2xl border border-(--color-paper-line) bg-(--color-paper-soft) text-(--color-ink)">
              <Bot size={24} strokeWidth={1.7} />
            </div>
            <h1 className="text-balance text-2xl font-semibold tracking-tight text-(--color-ink) sm:text-3xl">
              Apa yang ingin Anda ketahui?
            </h1>
            <p className="mx-auto mt-3 max-w-md text-sm leading-6 text-(--color-ink-soft)">
              Ajukan pertanyaan tentang dokumen. Jawaban akan merujuk ke halaman sumber.
            </p>
            {!disabled && (
              <div className="mt-7 flex flex-wrap justify-center gap-2">
                {SUGGESTED_QUESTIONS.map((question) => (
                  <button
                    key={question}
                    type="button"
                    onClick={() => askQuestion(question)}
                    disabled={isBusy || !sessionId}
                    className="rounded-full border border-(--color-paper-line) bg-(--color-paper-soft) px-4 py-2.5 text-sm text-(--color-ink-soft) transition-colors hover:bg-(--color-muted) hover:text-(--color-ink) focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--color-ink) disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {question}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {turns.map((turn, i) => (
          <article key={`${sessionId}-${i}`} className="mx-auto mb-9 flex w-full max-w-2xl flex-col gap-5 sm:mb-11">
            <div className="flex justify-end">
              <p className="max-w-[88%] rounded-3xl bg-(--color-muted) px-4 py-3 text-sm leading-6 text-(--color-ink) sm:max-w-[78%] sm:px-5">
                {turn.question}
              </p>
            </div>

            <div className="flex items-start gap-3">
              <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full border border-(--color-paper-line) bg-(--color-paper-soft) text-(--color-ink)">
                <Bot size={17} strokeWidth={1.8} />
              </div>
              <div className="min-w-0 flex-1 pt-2 items-center">
              {turn.error ? (
                <div className="rounded-xl border border-(--color-danger)/30 bg-(--color-danger)/5 px-4 py-3 text-sm leading-6 text-(--color-danger)" role="alert">
                  {turn.error}
                </div>
              ) : turn.isGenerating ? (
                <p className="flex gap-2 text-sm text-(--color-ink-soft)" role="status">
                  <span className="inline-flex gap-1">
                    <span className="size-1.5 animate-bounce rounded-full bg-(--color-ink-soft) [animation-delay:-0.3s] motion-reduce:animate-none" />
                    <span className="size-1.5 animate-bounce rounded-full bg-(--color-ink-soft) [animation-delay:-0.15s] motion-reduce:animate-none" />
                    <span className="size-1.5 animate-bounce rounded-full bg-(--color-ink-soft) motion-reduce:animate-none" />
                  </span>
                  {/* Sedang mencari jawaban... */}
                </p>
              ) : (
                <>
                  <div className="chat-markdown text-sm leading-6 text-(--color-ink)">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {turn.answer}
                    </ReactMarkdown>
                    {turn.isRevealing && (
                      <span className="ml-0.5 inline-block h-4 w-1 animate-pulse rounded-sm bg-(--color-ink-soft) align-middle motion-reduce:animate-none" />
                    )}
                  </div>

                  {turn.usedFallback && !turn.isRevealing && (
                    <p className="mt-3 text-xs text-(--color-ink-soft)">
                      Jawaban tidak ditemukan di dokumen.
                    </p>
                  )}

                  {!turn.isRevealing && turn.sources.length > 0 && (
                    <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-(--color-paper-line) pt-3">
                      <span className="mr-1 text-xs text-(--color-ink-soft)">Sumber</span>
                      {turn.sources.map((source, sourceIndex) => (
                        <span
                          key={`${source.file_name}-${source.page_label}-${sourceIndex}`}
                          title={source.file_name ?? undefined}
                          className="inline-flex items-center gap-1.5 rounded-full border border-(--color-paper-line) px-2.5 py-1 text-xs text-(--color-ink-soft)"
                        >
                          <FileText size={13} aria-hidden="true" />
                          Halaman {source.page_label ?? "?"}
                        </span>
                      ))}
                    </div>
                  )}
                </>
              )}
              </div>
            </div>
          </article>
        ))}
      </div>

      <div className="shrink-0 bg-(--color-background) px-1 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3 sm:px-2 sm:pb-5">
        <Form
          input={input}
          isBusy={isBusy}
          disabled={disabled || !sessionId}
          onSubmit={onSubmit}
          onInputChange={onInputChange}
        />
        <p className="mt-2 hidden text-center text-[11px] text-(--color-ink-soft) sm:block">
          Jawaban dibuat berdasarkan dokumen yang diunggah.
        </p>
      </div>
    </section>
  );
}
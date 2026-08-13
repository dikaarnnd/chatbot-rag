"use client";

import { useCallback, useEffect, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import { getSessionMessages, streamChat, type ChatEvent, type SourceCitation } from "@/lib/api";
import Form from "@/components/Form";

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
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const turnsLengthRef = useRef(0);

  useEffect(() => {
    turnsLengthRef.current = turns.length;
  }, [turns.length]);

  useEffect(() => {
    if (!sessionId) {
      setTurns([]);
      return;
    }

    let cancelled = false;
    setIsLoadingHistory(true);
    setTurns([]);

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

  const askQuestion = useCallback(async () => {
    const question = input.trim();
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

  const onInputChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  }, []);
 
  // ... (Pertahankan state dan logika fetching Anda yang sudah bagus)
  
  return (
    <div className="relative flex w-full max-w-3xl flex-col gap-4 pb-28">
      {/* Container pesan */}
      <div className="flex flex-1 flex-col gap-4">
        {isLoadingHistory && (
          <p className="text-center text-sm text-(--color-ink-soft)">Memuat riwayat percakapan...</p>
        )}

        {!isLoadingHistory && turns.length === 0 && (
          <p className="text-center text-sm text-(--color-ink-soft)">
            {disabled
              ? "Unggah dokumen terlebih dahulu untuk mulai bertanya."
              : "Ajukan pertanyaan tentang dokumen yang sudah diunggah."}
          </p>
        )}

        {turns.map((turn, i) => (
          <div key={i} className="rounded-lg border border-(--color-paper-line) bg-(--color-paper-soft) overflow-hidden shadow-sm">
            <div className="border-b border-(--color-paper-line) bg-(--color-accent-soft) px-4 py-3">
              <p className="font-serif text-sm font-medium text-(--color-ink) leading-relaxed">{turn.question}</p>
            </div>

            <div className="px-4 py-4">
               {/* ... (Pertahankan isi blok rendering teks & sources Anda) ... */}
            </div>
          </div>
        ))}
      </div>

      {/* Form Input Sticky Bottom */}
      <div className="fixed bottom-0 left-0 right-0 z-20 flex justify-center bg-gradient-to-t from-(--color-background) to-transparent p-4 md:left-[var(--sidebar-width,0px)]">
        <div className="w-full max-w-3xl">
          <Form
            input={input}
            isBusy={isBusy}
            disabled={disabled}
            onSubmit={onSubmit}
            onInputChange={onInputChange}
          />
        </div>
      </div>
    </div>
  );
}
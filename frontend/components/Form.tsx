"use client";

import { useEffect, useRef, type ChangeEvent, type FormEvent, type KeyboardEvent } from "react";
import { ArrowUp, LoaderCircle } from "lucide-react";

interface FormProps {
  input: string;
  isBusy: boolean;
  disabled: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onInputChange: (event: ChangeEvent<HTMLTextAreaElement>) => void;
  placeholder?: string;
}

export default function Form({
  input,
  isBusy,
  disabled,
  onSubmit,
  onInputChange,
  placeholder = "Tanyakan sesuatu tentang dokumen ini...",
}: FormProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }, [input]);

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <form
      onSubmit={onSubmit}
      className="w-full rounded-3xl border border-(--color-paper-line) bg-(--color-paper-soft) shadow-sm transition-shadow focus-within:border-(--color-ink-soft)/50 focus-within:shadow-md"
    >
      <div className="flex items-end gap-2 p-2 sm:p-2.5">
        <textarea
          ref={textareaRef}
          rows={1}
          value={input}
          onChange={(event) => {
            onInputChange(event);
          }}
          onKeyDown={handleKeyDown}
          disabled={isBusy || disabled}
          placeholder={disabled ? "Unggah dokumen terlebih dahulu..." : placeholder}
          aria-label="Tulis pertanyaan"
          className="max-h-40 min-h-11 min-w-0 flex-1 resize-none overflow-y-auto bg-transparent px-3 py-3 text-sm leading-5 text-(--color-ink) outline-none placeholder:text-(--color-ink-soft) disabled:cursor-not-allowed disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={isBusy || disabled || !input.trim()}
          aria-label={isBusy ? "Sedang memproses pertanyaan" : "Kirim pertanyaan"}
          title="Kirim"
          className="inline-flex size-10 shrink-0 items-center justify-center rounded-full bg-(--color-ink) text-(--color-background) transition-opacity hover:opacity-80 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--color-ink) disabled:cursor-not-allowed disabled:opacity-30"
        >
          {isBusy ? (
            <LoaderCircle className="animate-spin" size={18} />
          ) : (
            <ArrowUp size={19} strokeWidth={2.5} />
          )}
        </button>
      </div>
      {/* <p className="px-4 pb-2 text-center text-[11px] text-(--color-ink-soft)">
        Enter untuk mengirim · Shift + Enter untuk baris baru
      </p> */}
    </form>
  );
}
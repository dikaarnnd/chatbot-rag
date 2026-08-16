"use client";

import type { ChangeEvent, FormEvent } from "react";
import { Send, Loader } from 'lucide-react';

interface FormProps {
  input: string;
  isBusy: boolean;
  disabled: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onInputChange: (event: ChangeEvent<HTMLInputElement>) => void;
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
  return (
    <form
      onSubmit={onSubmit}
      className="flex w-full items-center gap-2 rounded-2xl border border-(--color-paper-line-strong) bg-(--color-paper-soft)/95 p-2 shadow-lg backdrop-blur"
    >
      <input
        type="text"
        value={input}
        onChange={onInputChange}
        disabled={isBusy || disabled}
        placeholder={disabled ? "Unggah dokumen terlebih dahulu..." : placeholder}
        className="flex-1 rounded-xl bg-transparent px-3 py-2 text-sm text-(--color-ink) placeholder:text-(--color-ink-soft) outline-none focus:ring-0 disabled:cursor-not-allowed disabled:opacity-60"
      />
      <button
        type="submit"
        disabled={isBusy || disabled || !input.trim()}
        className="shrink-0 rounded-xl bg-(--color-accent) px-4 py-2.5 text-sm font-medium text-black transition-opacity disabled:opacity-40 disabled:cursor-not-allowed hover:cursor-pointer"
      >
        {isBusy ? <Loader className="animate-spin" size={20}/> : <Send size={20} />}
      </button>
    </form>
  );
}
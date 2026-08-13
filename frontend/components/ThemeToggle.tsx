"use client";

import { useEffect, useState } from "react";
import { AiOutlineMoon, AiOutlineSun } from "react-icons/ai";

export default function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const current = document.documentElement.getAttribute("data-theme");
    setIsDark(current === "dark");
  }, []);

  const toggle = () => {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.setAttribute("data-theme", next ? "dark" : "light");
    localStorage.setItem("chatbotrag-theme", next ? "dark" : "light");
  };

  return (
    <button
      onClick={toggle}
      aria-label={isDark ? "Aktifkan mode terang" : "Aktifkan mode gelap"}
      className="rounded-full px-1.5 py-1.5 text-sm text-(--color-ink) transition-colors hover:bg-(--color-muted) not-dark:hover:bg-(--color-ink)"
    >
      {isDark ? <AiOutlineMoon size={18}/> : <AiOutlineSun size={18}/>}
    </button>
  );
}
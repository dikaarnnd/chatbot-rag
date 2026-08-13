"use client";

import { useState } from "react";
import {
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { PanelLeftOpen } from "lucide-react";

import AppSidebar from "@/components/AppSidebar";
import DocumentUpload from "@/components/DocumentUpload";
import ChatPanel from "@/components/ChatPanel";
import type { IngestResponse } from "@/lib/api";

function DesktopOpenTrigger() {
  const { open, toggleSidebar } = useSidebar();

  if (open) return null;

  return (
    <button
      type="button"
      onClick={toggleSidebar}
      className="hidden md:flex fixed top-3 left-3 z-40 p-2 rounded-md hover:bg-(--color-muted) text-(--color-ink) transition-colors border border-(--color-paper-line)"
      title="Buka Sidebar"
      aria-label="Buka Sidebar"
    >
      <PanelLeftOpen size={18} />
    </button>
  );
}

export default function Home() {
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [hasDocument, setHasDocument] = useState(false);
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0);

  const handleUploadSuccess = (result: IngestResponse) => {
    setActiveSessionId(result.session_id);
    setHasDocument(true);
    setSidebarRefreshKey((k) => k + 1);
  };

  const handleSelectSession = (sessionId: string) => {
    setActiveSessionId(sessionId);
    setHasDocument(true);
  };

  const handleNewChat = () => {
    setActiveSessionId(null);
    setHasDocument(false);
  };

  return (
    <>
      <AppSidebar
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        refreshKey={sidebarRefreshKey}
      />

      <main className="relative flex min-h-screen w-full flex-col">
        <DesktopOpenTrigger />

        <div className="sticky top-0 z-10 flex h-14 items-center border-b border-(--color-paper-line) bg-(--color-background) px-4 md:hidden">
          <SidebarTrigger
            type="button"
            className="-ml-2"
            aria-label="Buka Menu"
          />

          <span className="ml-3 font-serif text-lg font-medium text-(--color-ink)">
            Chatbot
          </span>
        </div>

        <div className="flex flex-1 flex-col items-center px-4 py-8 md:px-6 md:py-12">
          {!hasDocument ? (
            <div className="mx-auto mb-10 flex w-full max-w-2xl flex-col gap-6">
              <div className="text-center">
                <h1 className="font-serif text-2xl text-(--color-ink) sm:text-3xl">
                  Chatbot RAG
                </h1>

                <p className="mt-2 text-xs text-(--color-ink-soft) sm:text-sm">
                  Tanya-jawab grounded terhadap dokumen Modul Pembelajaran — jawaban selalu disertai halaman sumber.
                </p>
              </div>

              <DocumentUpload onUploadSuccess={handleUploadSuccess} />
            </div>
          ) : (
            <div className="flex w-full flex-col items-center">
              <ChatPanel
                sessionId={activeSessionId}
                disabled={!hasDocument}
              />
            </div>
          )}
        </div>
      </main>
    </>
  );
}
"use client";

import { useEffect, useState } from "react";
import { SquarePen, PanelLeftClose, MessageSquare } from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { listSessions, type SessionSummary } from "@/lib/api";

interface AppSidebarProps {
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
  refreshKey: number;
}

export default function AppSidebar({
  activeSessionId,
  onSelectSession,
  onNewChat,
  refreshKey,
}: AppSidebarProps) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { toggleSidebar, isMobile, setOpenMobile } = useSidebar();

  const handleClose = () => {
    if (isMobile) {
      setOpenMobile(false);
    } else {
      toggleSidebar();
    }
  };

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    
    listSessions()
      .then((data) => {
        if (!cancelled) setSessions(data);
      })
      .catch((err) => console.error("Gagal memuat riwayat chat:", err))
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  return (
    <Sidebar className="border-r border-(--color-paper-line)">
      <SidebarHeader className="px-3 pt-4">
        <div className="flex items-center justify-between pb-2">
          <span className="font-serif text-base font-semibold text-(--color-ink)">
            Chatbot RAG
          </span>
          {/* Tombol Tutup Sidebar Universal */}
          <button
            onClick={handleClose}
            className="p-1.5 rounded-md hover:bg-(--color-muted) text-(--color-ink-soft) hover:text-(--color-ink) transition-colors"
            title="Tutup Sidebar"
          >
            <PanelLeftClose size={18} />
          </button>
        </div>
        <Button 
          onClick={() => {
            onNewChat();
            if (isMobile) setOpenMobile(false); 
          }} 
          variant="outline" 
          className="w-full justify-start gap-2 shadow-none"
        >
          <SquarePen size={14} />
          <span className="text-sm">Percakapan Baru</span>
        </Button>
      </SidebarHeader>

      <SidebarContent className="px-1">
        <SidebarGroup>
          <SidebarGroupLabel className="text-(--color-ink-soft)">
            Riwayat Chat
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {isLoading && (
                <p className="px-2 py-2 text-xs text-(--color-ink-soft)">Memuat...</p>
              )}

              {!isLoading && sessions.length === 0 && (
                <p className="px-2 py-2 text-xs text-(--color-ink-soft)">
                  Belum ada riwayat dokumen.
                </p>
              )}

              {sessions.map((session) => {
                const isActive = session.id === activeSessionId;

                return (
                  <SidebarMenuItem key={session.id}>
                    <SidebarMenuButton
                      onClick={() => {
                        onSelectSession(session.id);
                        if (isMobile) setOpenMobile(false);
                      }}
                      isActive={isActive}
                      className={`gap-2 rounded-md transition-colors ${
                        isActive 
                          ? "bg-(--color-secondary) text-(--color-foreground)" 
                          : "text-(--color-muted-foreground) hover:bg-(--color-muted) hover:text-(--color-foreground)"
                      }`}
                    >
                      <MessageSquare className="shrink-0" size={16} />
                      <span className="truncate font-serif text-sm">{session.title}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
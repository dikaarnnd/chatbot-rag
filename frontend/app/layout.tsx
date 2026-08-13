import type { Metadata } from "next";
import { Geist, Geist_Mono, Source_Serif_4 } from "next/font/google";
import { SidebarProvider } from "@/components/ui/sidebar";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const sourceSerif = Source_Serif_4({
  variable: "--font-serif",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Chatbot RAG",
  description: "RAG-Based Chat Bot",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${sourceSerif.variable} h-full antialiased`} suppressHydrationWarning
    >
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                var stored = localStorage.getItem('chatbotrag-theme');
                var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                var isDark = stored ? stored === 'dark' : prefersDark;
                document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
                document.documentElement.classList.toggle('dark', isDark);
              })();
            `,
          }}
        />
      </head>
      <body className="min-h-full flex flex-col">
        <SidebarProvider>
          {children}
        </SidebarProvider>
      </body>
    </html>
  );
}

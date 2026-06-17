import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const plusJakarta = Plus_Jakarta_Sans({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["600", "700", "800"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "RoadGuard-X",
  description: "AI-powered offline road risk analysis — explainable computer vision",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en-GB" className="dark bg-[#05070A]">
      <body
        className={`${inter.variable} ${plusJakarta.variable} ${jetbrainsMono.variable} min-h-screen bg-[#05070A] font-sans text-zinc-100 antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

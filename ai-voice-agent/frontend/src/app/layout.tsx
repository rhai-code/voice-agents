import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Red Hat Pizza Shop",
  description: "AI Voice Agent - Red Hat Pizza Shop",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}

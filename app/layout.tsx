import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "iLumos — Patent intelligence workspace",
  description: "A focused workspace for refining patent claim charts.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

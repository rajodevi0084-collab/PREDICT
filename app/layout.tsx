import type { Metadata } from "next";
import "./styles/globals.css";

export const metadata: Metadata = {
  title: "Predict UI",
  description: "Frontend for managing dataset uploads, training runs, and predictions"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="app-shell">
          <header className="app-header">
            <h1 className="app-title">Predict Console</h1>
            <p className="app-subtitle">
              Manage datasets, training runs, and predictions from a single dashboard.
            </p>
          </header>
          <main className="app-main">{children}</main>
          <footer className="app-footer">
            Built with Next.js — configure the API endpoint via NEXT_PUBLIC_API_BASE_URL.
          </footer>
        </div>
      </body>
    </html>
  );
}

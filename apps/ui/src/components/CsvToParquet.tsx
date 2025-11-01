import { useState } from "react";

export function CsvToParquet() {
  const [symbol, setSymbol] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const fileInput = form.elements.namedItem("file") as HTMLInputElement;
    if (!fileInput.files || fileInput.files.length === 0) {
      setMessage("Select a CSV file first.");
      return;
    }
    const formData = new FormData();
    formData.append("symbol", symbol);
    formData.append("file", fileInput.files[0]);

    const response = await fetch("/utils/csv-to-parquet?symbol=" + encodeURIComponent(symbol), {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      setMessage("Conversion failed.");
      return;
    }
    const data = await response.json();
    setMessage(`Saved to ${data.path}`);
    form.reset();
  }

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Symbol
        <input value={symbol} onChange={(event) => setSymbol(event.target.value)} required />
      </label>
      <input type="file" name="file" accept=".csv" required />
      <button type="submit">Convert</button>
      {message && <p>{message}</p>}
    </form>
  );
}

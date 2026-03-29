import { useState, useCallback } from "react";
import Header from "@/components/Header";
import ImageInput from "@/components/ImageInput";
import OutputBox from "@/components/OutputBox";
import type { ResultStatus, DetectionResult } from "@/types";

export default function Home() {
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<ResultStatus>("idle");
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeNav, setActiveNav] = useState("Detection");

  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) return;

    setPreviewUrl(URL.createObjectURL(file));
    setStatus("loading");
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const text = await res.text();
      let payload: Record<string, unknown> = {};
      try {
        payload = text ? (JSON.parse(text) as Record<string, unknown>) : {};
      } catch {
        throw new Error(text?.slice(0, 200) || "Invalid response from server");
      }

      if (!res.ok) {
        const msg =
          (typeof payload.error === "string" && payload.error) ||
          (typeof payload.detail === "string" && payload.detail) ||
          `Analysis failed (${res.status})`;
        throw new Error(msg);
      }

      setResult(payload as DetectionResult);
      setStatus("success");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Something went wrong";
      setError(message);
      setStatus("error");
    }
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  const handleReset = () => {
    setPreviewUrl(null);
    setStatus("idle");
    setResult(null);
    setError(null);
  };

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ background: "linear-gradient(160deg, #f4f7f2 0%, #f0ede6 100%)" }}
    >
      <Header activeNav={activeNav} onNavChange={setActiveNav} />

      {/* Idle / Loading / Error — vertically centered */}
      {status !== "success" && (
        <main className="flex-1 flex flex-col items-center justify-center px-8 gap-8 w-full">
          <div className="w-full max-w-2xl flex flex-col items-center gap-8">
            <div className="text-center">
              <h1 className="text-[26px] font-bold text-[#111] tracking-tight leading-tight">
                Diagnose Your Apple Crop
              </h1>
              <p className="text-[14px] text-[#888] mt-2.5 max-w-md leading-relaxed">
                Upload a clear image of an apple leaf, fruit, or flower cluster. Our AI will detect diseases and provide actionable guidance.
              </p>
            </div>

            <ImageInput
              status={status}
              previewUrl={previewUrl}
              dragOver={dragOver}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={() => setDragOver(false)}
              onFileChange={handleFileChange}
              onReset={handleReset}
              onBrowseClick={() => {}}
            />

            {/* Error state */}
            {status === "error" && error && (
              <div
                className="w-full rounded-2xl px-5 py-4 text-[13px] text-[#c0392b] font-medium"
                style={{
                  background: "linear-gradient(135deg, #fff5f5, #fff0f0)",
                  border: "1px solid #f5c6c6",
                }}
              >
                ⚠ {error}
              </div>
            )}
          </div>
        </main>
      )}

      {/* Success — top-anchored two-column */}
      {status === "success" && result && previewUrl && (
        <main className="flex-1 flex flex-col items-center px-8 py-12 w-full">
          <div className="w-full max-w-5xl">
            <OutputBox result={result} previewUrl={previewUrl} onReset={handleReset} />
          </div>
        </main>
      )}
    </div>
  );
}
import { useState, useCallback } from "react";
import Header from "@/components/Header";
import ImageInput from "@/components/ImageInput";
import OutputBox from "@/components/OutputBox";
import { MOCK_RESULTS } from "@/constants";
import type { ResultStatus, DetectionResult } from "@/types";

export default function Home() {
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<ResultStatus>("idle");
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [activeNav, setActiveNav] = useState("Detection");

  const processFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    setPreviewUrl(URL.createObjectURL(file));
    setStatus("loading");
    setResult(null);
    // TODO: replace with real API call to /api/analyze
    setTimeout(() => {
      setResult(MOCK_RESULTS[Math.floor(Math.random() * MOCK_RESULTS.length)]);
      setStatus("success");
    }, 2200);
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
  };

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ background: "linear-gradient(160deg, #f4f7f2 0%, #f0ede6 100%)" }}
    >
      <Header activeNav={activeNav} onNavChange={setActiveNav} />

      {/* ── Idle / Loading: vertically centered like Claude ── */}
      {status !== "success" && (
        <main className="flex-1 flex flex-col items-center justify-center px-8 gap-8 w-full">
          <div className="w-full max-w-2xl flex flex-col items-center gap-8">

            {/* Title block */}
            <div className="text-center">
              <h1 className="text-[26px] font-bold text-[#111] tracking-tight leading-tight">
                Diagnose Your Apple Crop
              </h1>
              <p className="text-[14px] text-[#888] mt-2.5 max-w-md leading-relaxed">
                Upload a clear image of an apple leaf, fruit, or flower cluster. Our AI will detect diseases and provide actionable guidance.
              </p>
            </div>

            {/* Upload box */}
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
          </div>
        </main>
      )}

      {/* ── Success: top-anchored two-column layout ── */}
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
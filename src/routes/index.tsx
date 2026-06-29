import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { useServerFn } from "@tanstack/react-start";
import {
  Upload,
  Satellite,
  Cloud,
  Cpu,
  CheckCircle2,
  ShieldCheck,
  AlertTriangle,
  XCircle,
  Loader2,
  Activity,
  Download,
} from "lucide-react";
import { analyzeSatelliteImage } from "@/lib/satellite.functions";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "SatelliteVision AI — Cloud Reconstruction" },
      {
        name: "description",
        content:
          "Upload satellite imagery to detect clouds and reconstruct obscured terrain in real time.",
      },
      { property: "og:title", content: "SatelliteVision AI" },
      {
        property: "og:description",
        content: "Real-time satellite cloud detection and AI reconstruction.",
      },
    ],
  }),
  component: SatelliteVisionApp,
});

// ─── Types ────────────────────────────────────────────────
interface CloudRegion {
  xPercent: number;
  yPercent: number;
  widthPercent: number;
  heightPercent: number;
  density: number;
  shape: string;
}
interface ClaudeAnalysis {
  isSatelliteImage: boolean;
  satelliteConfidence: number;
  cloudPercentage: number;
  hasHighDensityClouds: boolean;
  cloudRegions: CloudRegion[];
  cloudThresholds: { brightnessMin: number; saturationMax: number };
  terrainFeatures: string[];
  terrainContext: {
    primaryLandUse: string;
    typicalColorR: number;
    typicalColorG: number;
    typicalColorB: number;
    textureComplexity: string;
  };
  notSatelliteReason: string | null;
  isDemoMode?: boolean;
}
interface PredictedFeature {
  class: string;
  confidence: number;
}
interface ProcessingResults {
  cloudMask: string;
  cloudOverlay: string;
  reconstructedImage: string;
  confidenceMap: string;
  cloudPercentage: number;
  processingTimeMs: number;
  inferenceTimeMs?: number;
  modelVersion?: string;
  terrainFeatures: string[];
  uncertaintyRegions: boolean;
  imageMetadata: {
    width: number;
    height: number;
    dominantColors: string[];
    isSatelliteImage: boolean;
    satelliteConfidence: number;
  };
  psnr: number;
  ssim: number;
  qualityReport?: string;
  reconstructionConfidence?: number;
  terrainClassificationMap: string;
  predictedFeatures?: PredictedFeature[];
}
interface LogEntry {
  timestamp: string;
  stage: string;
  message: string;
  duration?: number;
}
interface HistoryItem {
  id: string;
  filename: string;
  timestamp: string;
  uploadedImage: string;
  results: ProcessingResults;
}

// ─── Helpers ──────────────────────────────────────────────
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
async function resizeIfNeeded(file: File, maxPx = 1024): Promise<string> {
  const base64 = await fileToBase64(file);
  const img = await loadImage(base64);
  if (img.width <= maxPx && img.height <= maxPx) return base64;
  const scale = maxPx / Math.max(img.width, img.height);
  const c = document.createElement("canvas");
  c.width = Math.round(img.width * scale);
  c.height = Math.round(img.height * scale);
  c.getContext("2d")!.drawImage(img, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.92);
}

function extractDominantColors(pixels: ImageData): string[] {
  const buckets: Record<string, number> = {};
  for (let i = 0; i < pixels.data.length; i += 16) {
    const r = Math.floor(pixels.data[i] / 32) * 32;
    const g = Math.floor(pixels.data[i + 1] / 32) * 32;
    const b = Math.floor(pixels.data[i + 2] / 32) * 32;
    const k = `${r},${g},${b}`;
    buckets[k] = (buckets[k] || 0) + 1;
  }
  return Object.entries(buckets)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([k]) => {
      const [r, g, b] = k.split(",").map(Number);
      return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
    });
}

// ─── Component ────────────────────────────────────────────
const STAGES = [
  { label: "Upload", Icon: Upload },
  { label: "Validate", Icon: ShieldCheck },
  { label: "Cloud Detect", Icon: Cloud },
  { label: "Reconstruct", Icon: Cpu },
  { label: "Output", Icon: CheckCircle2 },
];

function nowTs() {
  const d = new Date();
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

function SatelliteVisionApp() {
  const analyze = useServerFn(analyzeSatelliteImage);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [originalFile, setOriginalFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);
  const [results, setResults] = useState<ProcessingResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [sliderPos, setSliderPos] = useState(50);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const addLog = useCallback((stage: string, message: string, duration?: number) => {
    setLog((prev) => {
      const next = [...prev, { timestamp: nowTs(), stage, message, duration }];
      return next.slice(-50);
    });
  }, []);

  const reset = useCallback(() => {
    setResults(null);
    setError(null);
    setCurrentStage(0);
    setLog([]);
    setSliderPos(50);
  }, []);

  // Load history on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem("cloudclear_history");
      if (saved) {
        setHistory(JSON.parse(saved));
      }
    } catch (e) {
      console.error("Failed to load history", e);
    }
  }, []);

  const loadHistoryItem = (item: HistoryItem) => {
    reset();
    setUploadedImage(item.uploadedImage);
    setResults(item.results);
    setCurrentStage(5);
    addLog("HISTORY", `Restored processed run: ${item.filename}`);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("cloudclear_history");
    addLog("HISTORY", "Run history cleared.");
  };

  const handleFile = useCallback(
    async (file: File): Promise<ProcessingResults | null> => {
      reset();
      setOriginalFile(file);
      setIsProcessing(true);
      const startTime = Date.now();
      try {
        addLog("UPLOAD", `Image received — ${file.name} (${Math.round(file.size / 1024)}KB)`);
        setCurrentStage(1);

        const isTiff = file.name.endsWith(".tif") || file.name.endsWith(".tiff");
        let base64Full: string;
        let imageData: string;
        let mediaType: string;

        if (isTiff) {
          addLog("UPLOAD", "TIFF/GeoTIFF format detected. Bypassing browser canvas downscaling.");
          base64Full = await fileToBase64(file);
          setUploadedImage(base64Full);
          const commaIdx = base64Full.indexOf(",");
          imageData = base64Full.substring(commaIdx + 1);
          mediaType = "image/tiff";
        } else {
          base64Full = await resizeIfNeeded(file, 1024);
          setUploadedImage(base64Full);
          const commaIdx = base64Full.indexOf(",");
          imageData = base64Full.substring(commaIdx + 1);
          mediaType = base64Full.substring(5, base64Full.indexOf(";"));
        }

        addLog("VALIDATE", "Uploading satellite image to server AI engine...");
        await new Promise((r) => setTimeout(r, 100));

        addLog("VALIDATE", "Initializing PyTorch Reconstruction Model on Server...");

        let response: Awaited<ReturnType<typeof analyze>>;
        try {
          response = await analyze({
            data: { imageBase64: imageData, mediaType },
          });
        } catch (e) {
          throw new Error(
            (e as Error)?.message || "AI analysis failed — please try a clearer satellite image",
          );
        }

        if (!response.isSatelliteImage) {
          throw new Error("NOT_SATELLITE");
        }

        // For TIFF uploads, the backend returns a browser-compatible preview
        let finalOriginalImage = response.originalImage || base64Full;
        setUploadedImage(finalOriginalImage);

        // Load preview in canvas to get metadata and colors
        const previewImg = await loadImage(finalOriginalImage);
        const previewCanvas = document.createElement("canvas");
        previewCanvas.width = previewImg.width;
        previewCanvas.height = previewImg.height;
        const previewCtx = previewCanvas.getContext("2d")!;
        previewCtx.drawImage(previewImg, 0, 0);
        const previewPixels = previewCtx.getImageData(0, 0, previewImg.width, previewImg.height);

        setCurrentStage(2);
        addLog(
          "CLOUD_DETECT",
          "Advanced Cloud segmentation: combining LAB, HSV, adaptive thresholds, and morphology",
        );
        addLog("CLOUD_DETECT", `Total cloud coverage estimated at ${response.cloudPercentage}%`);
        await new Promise((r) => setTimeout(r, 300));

        setCurrentStage(3);
        addLog(
          "RECONSTRUCT",
          "PyTorch Generative AI edge propagation & multi-scale blending active",
        );
        addLog(
          "RECONSTRUCT",
          `Terrain context: ${response.terrainContext.primaryLandUse} (${response.terrainContext.textureComplexity} texture complexity)`,
        );
        await new Promise((r) => setTimeout(r, 300));

        setCurrentStage(4);
        addLog(
          "CONFIDENCE",
          "Confidence mapping: distance transform and opacity analysis computed",
        );
        await new Promise((r) => setTimeout(r, 200));

        const elapsed = Date.now() - startTime;
        setCurrentStage(5);
        addLog(
          "OUTPUT",
          `AI reconstruction complete in ${response.processingTimeMs}ms (using ${response.deviceUsed})`,
          response.processingTimeMs,
        );

        const newResults: ProcessingResults = {
          cloudMask: response.cloudMask,
          cloudOverlay: response.cloudOverlay,
          reconstructedImage: response.reconstructedImage,
          confidenceMap: response.confidenceMap,
          cloudPercentage: response.cloudPercentage,
          processingTimeMs: response.processingTimeMs,
          inferenceTimeMs: response.inferenceTimeMs,
          modelVersion: response.modelVersion,
          terrainFeatures: response.terrainFeatures,
          uncertaintyRegions: response.hasHighDensityClouds,
          psnr: response.psnr,
          ssim: response.ssim,
          qualityReport: response.qualityReport,
          reconstructionConfidence: response.reconstructionConfidence,
          terrainClassificationMap: response.terrainClassificationMap,
          predictedFeatures: response.predictedFeatures,
          imageMetadata: {
            width: previewImg.width,
            height: previewImg.height,
            dominantColors: extractDominantColors(previewPixels),
            isSatelliteImage: true,
            satelliteConfidence: response.satelliteConfidence,
          },
        };

        setResults(newResults);

        // Add to run history
        const newHistoryItem: HistoryItem = {
          id: `${Date.now()}_${Math.random()}`,
          filename: file.name,
          timestamp: new Date().toLocaleTimeString(),
          uploadedImage: finalOriginalImage,
          results: newResults,
        };
        setHistory((prev) => {
          const updated = [newHistoryItem, ...prev].slice(0, 20);
          localStorage.setItem("cloudclear_history", JSON.stringify(updated));
          return updated;
        });

        return newResults;
      } catch (e) {
        const msg = (e as Error)?.message || "Unknown error";
        if (msg === "NOT_SATELLITE") {
          setError("NOT_SATELLITE");
          addLog("ERROR", "Image rejected: not a satellite image");
        } else {
          setError(msg);
          addLog("ERROR", msg);
        }
        setCurrentStage(0);
        return null;
      } finally {
        setIsProcessing(false);
      }
    },
    [addLog, analyze, reset],
  );

  const handleBatchFiles = useCallback(
    async (files: File[]) => {
      setIsProcessing(true);
      addLog("BATCH", `Starting batch processing of ${files.length} images...`);
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        addLog("BATCH", `[${i + 1}/${files.length}] Processing file: ${file.name}`);
        try {
          await handleFile(file);
        } catch (e) {
          addLog("BATCH", `Error processing ${file.name}: ${(e as Error).message}`);
        }
      }
      addLog("BATCH", "Batch processing complete.");
      setIsProcessing(false);
    },
    [handleFile, addLog],
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length > 0) {
      handleBatchFiles(files);
    }
  };

  return (
    <div className="min-h-screen">
      <Header />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <UploadZone
          dragActive={dragActive}
          setDragActive={setDragActive}
          uploadedImage={uploadedImage}
          isProcessing={isProcessing}
          onDrop={onDrop}
          onPick={() => inputRef.current?.click()}
        />
        <input
          ref={inputRef}
          type="file"
          accept="image/*,.tif,.tiff"
          multiple
          className="hidden"
          onChange={(e) => {
            const files = Array.from(e.target.files || []);
            if (files.length > 0) handleBatchFiles(files);
            e.target.value = "";
          }}
        />

        <PipelineBar currentStage={currentStage} isProcessing={isProcessing} />

        {error === "NOT_SATELLITE" && (
          <div className="mt-6 rounded-2xl border border-destructive/60 bg-destructive/10 p-5 text-destructive">
            <div className="flex items-center gap-3">
              <XCircle className="h-5 w-5" />
              <p className="text-display text-base font-semibold">
                This application only supports satellite imagery.
              </p>
            </div>
            <p className="mt-2 text-sm text-foreground/80">
              Please upload a top-down aerial or satellite image of Earth's surface.
            </p>
          </div>
        )}

        {error && error !== "NOT_SATELLITE" && (
          <div className="mt-6 rounded-2xl border border-warning/60 bg-warning/10 p-5">
            <div className="flex items-center gap-3 text-warning">
              <AlertTriangle className="h-5 w-5" />
              <p className="text-display text-base font-semibold">Processing failed</p>
            </div>
            <p className="mt-2 text-sm text-foreground/80">{error}</p>
            <button
              onClick={reset}
              className="text-mono mt-3 rounded-md border border-warning/60 px-3 py-1 text-xs text-warning hover:bg-warning/10"
            >
              Try Again
            </button>
          </div>
        )}

        {results && uploadedImage && (
          <ResultsGrid
            results={results}
            original={uploadedImage}
            sliderPos={sliderPos}
            setSliderPos={setSliderPos}
            originalFile={originalFile}
          />
        )}

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3 mt-6">
          <div className="md:col-span-1">
            <HistoryPanel history={history} onSelect={loadHistoryItem} onClear={clearHistory} />
          </div>
          <div className="md:col-span-2">
            <ProcessingLog log={log} isProcessing={isProcessing} />
          </div>
        </div>
      </main>
      <footer className="mx-auto max-w-7xl px-4 py-6 text-mono text-xs text-muted-foreground">
        SatelliteVision AI · pixel pipeline runs in your browser · vision inference via secure
        server
      </footer>
    </div>
  );
}

// ─── Subcomponents ────────────────────────────────────────
function Header() {
  return (
    <header className="border-b border-border/60 bg-surface/40 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-primary/15">
            <Satellite className="h-5 w-5 text-primary" />
            <span className="glow-cyan absolute inset-0 rounded-xl" />
          </div>
          <div>
            <h1 className="text-display text-lg font-bold tracking-tight">
              SatelliteVision <span className="text-primary">AI</span>
            </h1>
            <p className="text-mono text-[11px] text-muted-foreground">
              Real-Time Cloud Reconstruction &amp; Terrain Analysis
            </p>
          </div>
        </div>
        <span className="text-mono hidden rounded-full border border-secondary/40 bg-secondary/10 px-3 py-1 text-[11px] text-secondary-foreground/90 sm:inline-block">
          Powered by Claude Vision
        </span>
      </div>
    </header>
  );
}

function UploadZone({
  dragActive,
  setDragActive,
  uploadedImage,
  isProcessing,
  onDrop,
  onPick,
}: {
  dragActive: boolean;
  setDragActive: (v: boolean) => void;
  uploadedImage: string | null;
  isProcessing: boolean;
  onDrop: (e: React.DragEvent) => void;
  onPick: () => void;
}) {
  return (
    <section
      onDragOver={(e) => {
        e.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={onDrop}
      className={`relative rounded-2xl border-2 border-dashed bg-surface/30 p-8 transition-colors ${
        dragActive ? "border-primary bg-primary/5" : "border-border"
      }`}
    >
      <div className="flex flex-col items-center justify-center gap-3 text-center">
        <div className="relative">
          {isProcessing ? (
            <Loader2 className="animate-spin-slow h-10 w-10 text-primary" />
          ) : (
            <Satellite className="h-10 w-10 text-primary" />
          )}
        </div>
        <h2 className="text-display text-xl font-semibold">Drop satellite imagery here</h2>
        <p className="text-mono text-xs text-muted-foreground">
          Supports: GeoTIFF · Sentinel-2 · Landsat · ISRO LISS-IV · PNG · JPG
        </p>
        <button
          onClick={onPick}
          disabled={isProcessing}
          className="mt-2 rounded-md border border-primary/60 bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition hover:bg-primary/20 disabled:opacity-50"
        >
          {isProcessing ? "Processing..." : "Select Image"}
        </button>
        <p className="text-mono text-[10px] text-muted-foreground">
          max 4MB · downscaled to 1024px on longest side
        </p>

        {uploadedImage && (
          <div className="mt-4 overflow-hidden rounded-lg border border-border">
            <img
              src={uploadedImage}
              alt="Uploaded preview"
              className="max-h-40 w-auto object-contain"
            />
          </div>
        )}
      </div>
    </section>
  );
}

function PipelineBar({
  currentStage,
  isProcessing,
}: {
  currentStage: number;
  isProcessing: boolean;
}) {
  return (
    <section className="mt-6 rounded-2xl border border-border bg-surface/40 p-5">
      <div className="mb-3 flex items-center justify-between">
        <p className="text-mono text-[11px] uppercase tracking-widest text-muted-foreground">
          Processing Pipeline
        </p>
        <p className="text-mono text-[11px] text-muted-foreground">
          {isProcessing ? "ACTIVE" : currentStage >= 5 ? "COMPLETE" : "IDLE"}
        </p>
      </div>
      <div className="flex items-center">
        {STAGES.map((s, i) => {
          const isDone = currentStage > i;
          const isActive = currentStage === i + 1 && isProcessing;
          const Icon = s.Icon;
          return (
            <div key={s.label} className="flex flex-1 items-center">
              <div className="flex flex-col items-center">
                <div
                  className={`flex h-11 w-11 items-center justify-center rounded-full border transition-all ${
                    isDone
                      ? "border-success bg-success/20 text-success"
                      : isActive
                        ? "animate-pulse-glow border-primary bg-primary/20 text-primary"
                        : "border-border bg-surface text-muted-foreground"
                  }`}
                >
                  <Icon className="h-5 w-5" />
                </div>
                <span className="text-mono mt-2 text-[10px] uppercase tracking-wider text-muted-foreground">
                  {s.label}
                </span>
              </div>
              {i < STAGES.length - 1 && (
                <div className="mx-2 h-px flex-1 relative overflow-hidden">
                  <div className={`absolute inset-0 ${isDone ? "bg-success/60" : "bg-border"}`} />
                  {isActive && <div className="animate-flow absolute inset-0" />}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function HistoryPanel({
  history,
  onSelect,
  onClear,
}: {
  history: HistoryItem[];
  onSelect: (item: HistoryItem) => void;
  onClear: () => void;
}) {
  return (
    <section className="h-full overflow-hidden rounded-2xl border border-border bg-surface/60 flex flex-col">
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-2 bg-surface/40">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          <span className="text-display text-sm font-semibold">Run History</span>
        </div>
        {history.length > 0 && (
          <button
            onClick={onClear}
            className="text-mono text-[10px] text-destructive hover:underline"
          >
            Clear All
          </button>
        )}
      </div>
      <div className="max-h-64 overflow-y-auto bg-background/60 p-3 flex-1">
        {history.length === 0 ? (
          <p className="text-mono text-xs text-muted-foreground text-center py-8">
            No history yet. Process an image to save.
          </p>
        ) : (
          <div className="space-y-2">
            {history.map((item) => (
              <button
                key={item.id}
                onClick={() => onSelect(item)}
                className="w-full flex items-center gap-3 p-2 rounded-lg border border-border/50 bg-surface/40 hover:bg-surface/80 hover:border-primary/50 text-left transition"
              >
                <div className="h-10 w-10 overflow-hidden rounded border border-border flex-shrink-0 bg-background">
                  <img
                    src={item.results.reconstructedImage}
                    alt="Reconstructed thumb"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-display text-xs font-semibold truncate text-foreground">
                    {item.filename}
                  </p>
                  <p className="text-mono text-[9px] text-muted-foreground flex justify-between">
                    <span>{item.timestamp}</span>
                    <span className="text-primary font-bold">{item.results.cloudPercentage}% cloud</span>
                  </p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function downloadImage(base64Data: string, filename: string) {
  const link = document.createElement("a");
  link.href = base64Data;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function ResultsGrid({
  results,
  original,
  sliderPos,
  setSliderPos,
  originalFile,
}: {
  results: ProcessingResults;
  original: string;
  sliderPos: number;
  setSliderPos: (n: number) => void;
  originalFile: File | null;
}) {
  return (
    <section className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Panel
        title="Original Satellite Image"
        badge={`${results.imageMetadata.width}×${results.imageMetadata.height}px · ${originalFile?.type?.split("/")[1]?.toUpperCase() || "IMG"}`}
        delay={0}
        topRight={
          <button
            onClick={() => downloadImage(original, "original.png")}
            className="text-mono rounded-md border border-primary/60 bg-primary/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-primary hover:bg-primary/20 flex items-center gap-1"
          >
            <Download className="h-3 w-3" />
            Download
          </button>
        }
      >
        <img src={original} alt="Original" className="h-full w-full object-cover" />
      </Panel>

      <Panel
        title="Detected Cloud Mask"
        badge={`Cloud Coverage: ${results.cloudPercentage}%`}
        delay={100}
        topRight={
          <button
            onClick={() => downloadImage(results.cloudMask, "cloud_mask.png")}
            className="text-mono rounded-md border border-primary/60 bg-primary/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-primary hover:bg-primary/20 flex items-center gap-1"
          >
            <Download className="h-3 w-3" />
            Download
          </button>
        }
      >
        <img src={results.cloudMask} alt="Cloud mask" className="h-full w-full object-cover" />
      </Panel>

      <Panel
        title="AI Reconstructed Image"
        badge={results.terrainFeatures.slice(0, 2).join(" · ") || "terrain"}
        delay={200}
        topRight={
          <div className="flex items-center gap-2">
            <button
              onClick={() => downloadImage(results.reconstructedImage, "reconstructed.png")}
              className="text-mono rounded-md border border-primary/60 bg-primary/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-primary hover:bg-primary/20 flex items-center gap-1"
            >
              <Download className="h-3 w-3" />
              Download
            </button>
            <span className="text-mono rounded-md border border-success/60 bg-success/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-success">
              Reconstructed
            </span>
          </div>
        }
      >
        <img
          src={results.reconstructedImage}
          alt="Reconstructed"
          className="h-full w-full object-cover"
        />
      </Panel>

      <Panel title="Before / After Comparison" delay={300}>
        <ComparisonSlider
          left={original}
          right={results.reconstructedImage}
          pos={sliderPos}
          setPos={setSliderPos}
        />
      </Panel>

      <Panel
        title="Reconstruction Confidence"
        delay={400}
        topRight={
          <button
            onClick={() => downloadImage(results.confidenceMap, "confidence_map.png")}
            className="text-mono rounded-md border border-primary/60 bg-primary/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-primary hover:bg-primary/20 flex items-center gap-1"
          >
            <Download className="h-3 w-3" />
            Download
          </button>
        }
      >
        <div className="animate-reveal h-full w-full">
          <img
            src={results.confidenceMap}
            alt="Confidence"
            className="h-full w-full object-cover"
          />
        </div>
        <div className="absolute inset-x-3 bottom-3 rounded-md bg-background/80 p-2 backdrop-blur">
          <div className="h-2 w-full rounded-full bg-gradient-to-r from-destructive via-warning to-success" />
          <div className="text-mono mt-1 flex justify-between text-[9px] uppercase tracking-wider text-muted-foreground">
            <span>Low Confidence</span>
            <span>High Confidence</span>
          </div>
        </div>
      </Panel>

      <Panel
        title="Terrain Classification Map"
        delay={450}
        topRight={
          <button
            onClick={() => downloadImage(results.terrainClassificationMap, "terrain_classification.png")}
            className="text-mono rounded-md border border-primary/60 bg-primary/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-primary hover:bg-primary/20 flex items-center gap-1"
          >
            <Download className="h-3 w-3" />
            Download
          </button>
        }
      >
        <div className="animate-reveal h-full w-full">
          <img
            src={results.terrainClassificationMap}
            alt="Terrain Classification"
            className="h-full w-full object-cover"
          />
        </div>
        <div className="absolute inset-x-3 bottom-3 rounded-md bg-background/80 p-2 backdrop-blur">
          <div className="text-mono flex flex-wrap gap-x-2 gap-y-1 text-[8px] uppercase tracking-wider text-muted-foreground justify-center">
            <span className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-[#b41e1e]" />
              Urban Area
            </span>
            <span className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-[#f05050]" />
              Buildings
            </span>
            <span className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-[#808080]" />
              Roads
            </span>
            <span className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-[#228b22]" />
              Forest
            </span>
            <span className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-[#7cfc00]" />
              Agri
            </span>
            <span className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-[#0000ff]" />
              Water
            </span>
            <span className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-[#8b4513]" />
              Mountain
            </span>
            <span className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-[#d2b48c]" />
              Desert
            </span>
            <span className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-[#8e6b23]" />
              Bare Land
            </span>
          </div>
        </div>
      </Panel>

      <Panel title="Analysis Report" delay={500}>
        <div className="grid h-full grid-cols-1 gap-2 p-4 text-sm">
          <Stat label="☁️ Cloud Coverage" value={`${results.cloudPercentage}%`} />
          <Stat label="⏱ Total Process Time" value={`${results.processingTimeMs}ms`} />
          {results.inferenceTimeMs !== undefined && (
            <Stat label="⚡️ Model Inference Time" value={`${results.inferenceTimeMs}ms`} />
          )}
          {results.modelVersion && (
            <Stat label="🤖 Reconstruction Model" value={results.modelVersion} />
          )}
          <Stat label="🎯 PSNR (Peak SNR)" value={`${results.psnr} dB`} />
          <Stat label="⚖️ SSIM (Structural Sim)" value={results.ssim.toFixed(4)} />
          <Stat
            label="📐 Resolution"
            value={`${results.imageMetadata.width}×${results.imageMetadata.height}px`}
          />
          <Stat label="🌍 Terrain" value={results.terrainFeatures.join(", ") || "—"} />
          <Stat
            label="🎯 Satellite Confidence"
            value={`${results.imageMetadata.satelliteConfidence}%`}
          />
          <div className="mt-1 flex flex-wrap gap-1">
            {results.imageMetadata.dominantColors.map((c) => (
              <span
                key={c}
                title={c}
                className="h-4 w-4 rounded-sm border border-border"
                style={{ backgroundColor: c }}
              />
            ))}
          </div>
          {results.qualityReport && (
            <div
              className={`text-mono mt-2 rounded-md border p-2 text-[11px] ${
                results.qualityReport.includes("Poor")
                  ? "border-warning/50 bg-warning/10 text-warning"
                  : "border-primary/50 bg-primary/10 text-primary"
              }`}
            >
              {results.qualityReport.includes("Poor") ? "⚠️" : "✨"} {results.qualityReport}
            </div>
          )}
          {results.uncertaintyRegions && !results.qualityReport?.includes("Poor") && (
            <div className="text-mono mt-2 rounded-md border border-warning/50 bg-warning/10 p-2 text-[11px] text-warning">
              ⚠️ Uncertainty: dense cloud cover detected. The optical image does not contain enough
              information beneath the cloud to fully reconstruct this region.
            </div>
          )}
        </div>
      </Panel>

      <Panel title="AI Predicted Hidden Terrain" delay={550}>
        <div className="grid h-full grid-cols-1 gap-3 p-4 text-sm overflow-y-auto">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider text-mono">
            ⚠️ Most Probable Features Estimated Under Cloud Cover
          </p>
          {!results.predictedFeatures || results.predictedFeatures.length === 0 ? (
            <p className="text-mono text-xs text-muted-foreground py-6 text-center">
              No clouds detected. No hidden terrain estimation needed.
            </p>
          ) : (
            <div className="space-y-3">
              {results.predictedFeatures.map((item) => (
                <div key={item.class} className="space-y-1">
                  <div className="flex justify-between text-xs text-mono">
                    <span className="font-semibold text-foreground">{item.class}</span>
                    <span className="text-primary font-bold">{item.confidence}% Estimated</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-border/40 overflow-hidden relative">
                    <div
                      className="absolute inset-y-0 left-0 bg-primary rounded-full transition-all duration-500"
                      style={{ width: `${item.confidence}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </Panel>
    </section>
  );
}

function Panel({
  title,
  badge,
  topRight,
  children,
  delay = 0,
}: {
  title: string;
  badge?: string;
  topRight?: React.ReactNode;
  children: React.ReactNode;
  delay?: number;
}) {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setShown(true), delay);
    return () => clearTimeout(t);
  }, [delay]);
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-border bg-surface/60 transition-opacity duration-500 ${
        shown ? "opacity-100" : "opacity-0"
      }`}
    >
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-2">
        <h3 className="text-display text-sm font-semibold">{title}</h3>
        {topRight}
      </div>
      <div className="relative aspect-square w-full bg-background">
        {children}
        {badge && (
          <span className="text-mono absolute left-3 top-3 rounded-md bg-background/80 px-2 py-1 text-[10px] uppercase tracking-wider text-primary backdrop-blur">
            {badge}
          </span>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-border/40 py-1.5">
      <span className="text-mono text-[11px] text-muted-foreground">{label}</span>
      <span className="text-display text-sm font-medium">{value}</span>
    </div>
  );
}

function ComparisonSlider({
  left,
  right,
  pos,
  setPos,
}: {
  left: string;
  right: string;
  pos: number;
  setPos: (n: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const move = useCallback(
    (clientX: number) => {
      const el = containerRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const x = Math.min(Math.max(clientX - r.left, 0), r.width);
      requestAnimationFrame(() => setPos((x / r.width) * 100));
    },
    [setPos],
  );

  useEffect(() => {
    const up = () => (draggingRef.current = false);
    const mv = (e: MouseEvent) => draggingRef.current && move(e.clientX);
    const tm = (e: TouchEvent) => draggingRef.current && move(e.touches[0]?.clientX ?? 0);
    window.addEventListener("mouseup", up);
    window.addEventListener("mousemove", mv);
    window.addEventListener("touchmove", tm);
    window.addEventListener("touchend", up);
    return () => {
      window.removeEventListener("mouseup", up);
      window.removeEventListener("mousemove", mv);
      window.removeEventListener("touchmove", tm);
      window.removeEventListener("touchend", up);
    };
  }, [move]);

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full select-none overflow-hidden"
      onMouseDown={(e) => {
        draggingRef.current = true;
        move(e.clientX);
      }}
      onTouchStart={(e) => {
        draggingRef.current = true;
        move(e.touches[0]?.clientX ?? 0);
      }}
    >
      <img
        src={right}
        alt="Reconstructed"
        className="absolute inset-0 h-full w-full object-cover"
      />
      <div className="absolute inset-y-0 left-0 overflow-hidden" style={{ width: `${pos}%` }}>
        <img
          src={left}
          alt="Original"
          className="absolute inset-0 h-full w-full object-cover"
          style={{ width: `${(100 / pos) * 100}%`, maxWidth: "none" }}
        />
      </div>
      <span className="text-mono absolute left-2 top-2 rounded bg-background/70 px-2 py-0.5 text-[10px] uppercase tracking-wider text-primary backdrop-blur">
        Original
      </span>
      <span className="text-mono absolute right-2 top-2 rounded bg-background/70 px-2 py-0.5 text-[10px] uppercase tracking-wider text-success backdrop-blur">
        Reconstructed
      </span>
      <div className="absolute inset-y-0 w-px bg-primary" style={{ left: `${pos}%` }}>
        <div className="glow-cyan absolute top-1/2 -translate-x-1/2 -translate-y-1/2 flex h-9 w-9 cursor-ew-resize items-center justify-center rounded-full border border-primary bg-background text-primary">
          <span className="text-mono text-xs">◀▶</span>
        </div>
      </div>
    </div>
  );
}

function ProcessingLog({ log, isProcessing }: { log: LogEntry[]; isProcessing: boolean }) {
  const [open, setOpen] = useState(true);
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [log.length]);

  return (
    <section className="mt-6 overflow-hidden rounded-2xl border border-border bg-surface/60">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between border-b border-border/60 px-4 py-2"
      >
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          <span className="text-display text-sm font-semibold">Processing Log</span>
          {isProcessing && (
            <span className="ml-2 inline-block h-2 w-2 animate-pulse rounded-full bg-success" />
          )}
        </div>
        <span className="text-mono text-[11px] text-muted-foreground">
          {log.length} entries · {open ? "hide" : "show"}
        </span>
      </button>
      {open && (
        <div className="max-h-64 overflow-y-auto bg-background/60 p-3">
          {log.length === 0 ? (
            <p className="text-mono text-xs text-muted-foreground">
              Awaiting input. Upload a satellite image to begin.
            </p>
          ) : (
            <ul className="text-mono space-y-1 text-[11px]">
              {log.map((e, i) => (
                <li key={i} className="animate-slide-in-left flex items-start gap-2">
                  <span className="text-muted-foreground">[{e.timestamp}]</span>
                  <span className="text-primary">{e.stage}</span>
                  <span className="text-muted-foreground">→</span>
                  <span className="flex-1 text-foreground/90">{e.message}</span>
                  {e.duration !== undefined && (
                    <span className="text-success">({e.duration}ms)</span>
                  )}
                </li>
              ))}
            </ul>
          )}
          <div ref={endRef} />
        </div>
      )}
    </section>
  );
}

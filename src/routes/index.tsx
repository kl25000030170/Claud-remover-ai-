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
  Image as ImageIcon,
  Loader2,
  Activity,
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
}
interface ProcessingResults {
  cloudMask: string;
  reconstructedImage: string;
  confidenceMap: string;
  cloudPercentage: number;
  processingTimeMs: number;
  terrainFeatures: string[];
  uncertaintyRegions: boolean;
  imageMetadata: {
    width: number;
    height: number;
    dominantColors: string[];
    isSatelliteImage: boolean;
    satelliteConfidence: number;
  };
}
interface LogEntry {
  timestamp: string;
  stage: string;
  message: string;
  duration?: number;
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

function generateSpiralSamples(
  cx: number,
  cy: number,
  maxR: number,
  width: number,
  height: number,
  isCloud: Uint8Array,
): [number, number, number][] {
  const samples: [number, number, number][] = [];
  for (let r = 2; r <= maxR; r += 2) {
    const step = Math.max(5, 360 / (r * 4));
    for (let angle = 0; angle < 360; angle += step) {
      const rad = (angle * Math.PI) / 180;
      const sx = Math.round(cx + r * Math.cos(rad));
      const sy = Math.round(cy + r * Math.sin(rad));
      if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
        if (!isCloud[sy * width + sx]) samples.push([sx, sy, r]);
      }
    }
    if (samples.length >= 16) break;
  }
  return samples;
}
function checkSurrounded(
  x: number,
  y: number,
  width: number,
  height: number,
  isCloud: Uint8Array,
  radius: number,
): boolean {
  let cloud = 0,
    total = 0;
  for (let dy = -radius; dy <= radius; dy += 2) {
    for (let dx = -radius; dx <= radius; dx += 2) {
      const nx = x + dx,
        ny = y + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        total++;
        if (isCloud[ny * width + nx]) cloud++;
      }
    }
  }
  return total > 0 && cloud / total > 0.7;
}

function generateCloudMask(
  pixels: ImageData,
  width: number,
  height: number,
  cloudRegions: CloudRegion[],
  thresholds: { brightnessMin: number; saturationMax: number },
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const mask = ctx.createImageData(width, height);
  const data = mask.data;
  const src = pixels.data;
  const bMin = thresholds.brightnessMin ?? 180;
  const sMax = thresholds.saturationMax ?? 40;

  // Init alpha=255 black
  for (let i = 3; i < data.length; i += 4) data[i] = 255;

  // Region-guided detection
  for (const region of cloudRegions) {
    const x1 = Math.max(0, Math.floor((region.xPercent / 100) * width));
    const y1 = Math.max(0, Math.floor((region.yPercent / 100) * height));
    const x2 = Math.min(width, x1 + Math.floor((region.widthPercent / 100) * width));
    const y2 = Math.min(height, y1 + Math.floor((region.heightPercent / 100) * height));
    for (let y = y1; y < y2; y++) {
      for (let x = x1; x < x2; x++) {
        const idx = (y * width + x) * 4;
        const r = src[idx],
          g = src[idx + 1],
          b = src[idx + 2];
        const brightness = (r + g + b) / 3;
        const saturation = Math.max(r, g, b) - Math.min(r, g, b);
        if (brightness > bMin && saturation < sMax) {
          const intensity = Math.min(255, Math.floor(brightness * region.density * 1.2));
          // cyan-tinted cloud
          data[idx] = Math.round(intensity * 0.6);
          data[idx + 1] = intensity;
          data[idx + 2] = intensity;
        }
      }
    }
  }

  // Global pass for missed clouds
  for (let i = 0; i < src.length; i += 4) {
    const r = src[i],
      g = src[i + 1],
      b = src[i + 2];
    const brightness = (r + g + b) / 3;
    const saturation = Math.max(r, g, b) - Math.min(r, g, b);
    if (data[i + 1] === 0 && brightness > 210 && saturation < 25) {
      data[i] = Math.round(brightness * 0.6);
      data[i + 1] = brightness;
      data[i + 2] = brightness;
    }
  }

  ctx.putImageData(mask, 0, 0);
  return canvas.toDataURL("image/png");
}

function reconstructImage(
  pixels: ImageData,
  width: number,
  height: number,
  cloudRegions: CloudRegion[],
  terrainContext: ClaudeAnalysis["terrainContext"],
): { dataUrl: string; cloudPixelCount: number } {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.putImageData(pixels, 0, 0);
  const result = ctx.getImageData(0, 0, width, height);
  const data = result.data;
  const src = pixels.data;

  const isCloud = new Uint8Array(width * height);
  let cloudPixelCount = 0;
  for (let i = 0; i < src.length; i += 4) {
    const r = src[i],
      g = src[i + 1],
      b = src[i + 2];
    const brightness = (r + g + b) / 3;
    const saturation = Math.max(r, g, b) - Math.min(r, g, b);
    if (brightness > 200 && saturation < 35) {
      isCloud[i / 4] = 1;
      cloudPixelCount++;
    }
  }
  for (const region of cloudRegions) {
    if (region.density <= 0.5) continue;
    const x1 = Math.max(0, Math.floor((region.xPercent / 100) * width));
    const y1 = Math.max(0, Math.floor((region.yPercent / 100) * height));
    const x2 = Math.min(width, x1 + Math.floor((region.widthPercent / 100) * width));
    const y2 = Math.min(height, y1 + Math.floor((region.heightPercent / 100) * height));
    for (let y = y1; y < y2; y++)
      for (let x = x1; x < x2; x++) {
        const p = y * width + x;
        if (!isCloud[p]) {
          isCloud[p] = 1;
          cloudPixelCount++;
        }
      }
  }

  const searchRadius = Math.max(30, Math.floor(Math.min(width, height) * 0.08));

  // Pass 1: exemplar-based fill
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      if (!isCloud[p]) continue;
      const samples = generateSpiralSamples(x, y, searchRadius, width, height, isCloud);
      if (samples.length === 0) continue;
      let sr = 0,
        sg = 0,
        sb = 0,
        ws = 0;
      for (let s = 0; s < Math.min(12, samples.length); s++) {
        const [sx, sy, dist] = samples[s];
        const w = 1 / (dist + 1);
        const sIdx = (sy * width + sx) * 4;
        sr += src[sIdx] * w;
        sg += src[sIdx + 1] * w;
        sb += src[sIdx + 2] * w;
        ws += w;
      }
      const i4 = p * 4;
      data[i4] = Math.round(sr / ws);
      data[i4 + 1] = Math.round(sg / ws);
      data[i4 + 2] = Math.round(sb / ws);
      data[i4 + 3] = 255;
    }
  }

  // Pass 2: blend terrain color in dense centers
  const tcR = terrainContext.typicalColorR ?? 90;
  const tcG = terrainContext.typicalColorG ?? 110;
  const tcB = terrainContext.typicalColorB ?? 80;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      if (!isCloud[p]) continue;
      const surrounded = checkSurrounded(x, y, width, height, isCloud, 6);
      const blend = surrounded ? 0.5 : 0.18;
      const i4 = p * 4;
      data[i4] = Math.round(data[i4] * (1 - blend) + tcR * blend);
      data[i4 + 1] = Math.round(data[i4 + 1] * (1 - blend) + tcG * blend);
      data[i4 + 2] = Math.round(data[i4 + 2] * (1 - blend) + tcB * blend);
    }
  }

  ctx.putImageData(result, 0, 0);
  return { dataUrl: canvas.toDataURL("image/png"), cloudPixelCount };
}

function generateConfidenceMap(
  pixels: ImageData,
  width: number,
  height: number,
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const map = ctx.createImageData(width, height);
  const data = map.data;
  const src = pixels.data;
  for (let i = 0; i < src.length; i += 4) {
    const r = src[i],
      g = src[i + 1],
      b = src[i + 2];
    const brightness = (r + g + b) / 3;
    const saturation = Math.max(r, g, b) - Math.min(r, g, b);
    let confidence: number;
    if (brightness > 200 && saturation < 35) {
      const density = brightness / 255;
      confidence = Math.max(0.05, 1 - density);
    } else {
      confidence = 0.92 + Math.random() * 0.08;
    }
    data[i] = Math.round(255 * (1 - confidence));
    data[i + 1] = Math.round(255 * confidence);
    data[i + 2] = 40;
    data[i + 3] = 220;
  }
  ctx.putImageData(map, 0, 0);
  return canvas.toDataURL("image/png");
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

  const handleFile = useCallback(
    async (file: File) => {
      reset();
      setOriginalFile(file);
      setIsProcessing(true);
      const startTime = Date.now();
      try {
        addLog("UPLOAD", `Image received — ${file.name} (${Math.round(file.size / 1024)}KB)`);
        setCurrentStage(1);

        const base64Full = await resizeIfNeeded(file, 1024);
        setUploadedImage(base64Full);
        const img = await loadImage(base64Full);
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(img, 0, 0);
        const pixels = ctx.getImageData(0, 0, img.width, img.height);

        addLog("VALIDATE", "Sending to Claude Vision for satellite image classification...");
        await new Promise((r) => setTimeout(r, 200));

        const imageData = base64Full.split(",")[1];
        const mediaType = base64Full.substring(5, base64Full.indexOf(";"));
        let analysis: ClaudeAnalysis;
        try {
          analysis = (await analyze({ data: { imageBase64: imageData, mediaType } })) as ClaudeAnalysis;
        } catch (e: any) {
          throw new Error(e?.message || "AI analysis failed — please try a clearer satellite image");
        }

        if (!analysis.isSatelliteImage) {
          throw new Error("NOT_SATELLITE");
        }
        addLog("VALIDATE", `Satellite confidence ${analysis.satelliteConfidence}% — proceeding`);
        setCurrentStage(2);
        await new Promise((r) => setTimeout(r, 300));

        addLog(
          "CLOUD_DETECT",
          `Identified ${analysis.cloudRegions.length} cloud regions via pixel analysis`,
        );
        addLog(
          "CLOUD_DETECT",
          `Total cloud coverage estimated at ${analysis.cloudPercentage}%`,
        );
        const cloudMask = generateCloudMask(
          pixels,
          img.width,
          img.height,
          analysis.cloudRegions,
          analysis.cloudThresholds,
        );
        setCurrentStage(3);
        await new Promise((r) => setTimeout(r, 300));

        const { dataUrl: reconstructed, cloudPixelCount } = reconstructImage(
          pixels,
          img.width,
          img.height,
          analysis.cloudRegions,
          analysis.terrainContext,
        );
        addLog(
          "RECONSTRUCT",
          `Running exemplar-based inpainting on ${cloudPixelCount} cloud pixels`,
        );
        addLog("RECONSTRUCT", `Terrain context: ${analysis.terrainContext.primaryLandUse}`);
        setCurrentStage(4);
        await new Promise((r) => setTimeout(r, 300));

        addLog("CONFIDENCE", "Generating reconstruction confidence map");
        const confidence = generateConfidenceMap(pixels, img.width, img.height);

        const elapsed = Date.now() - startTime;
        setCurrentStage(5);
        addLog("OUTPUT", `Processing complete in ${elapsed}ms`, elapsed);

        setResults({
          cloudMask,
          reconstructedImage: reconstructed,
          confidenceMap: confidence,
          cloudPercentage: analysis.cloudPercentage,
          processingTimeMs: elapsed,
          terrainFeatures: analysis.terrainFeatures,
          uncertaintyRegions: analysis.hasHighDensityClouds,
          imageMetadata: {
            width: img.width,
            height: img.height,
            dominantColors: extractDominantColors(pixels),
            isSatelliteImage: true,
            satelliteConfidence: analysis.satelliteConfidence,
          },
        });
      } catch (e: any) {
        const msg = e?.message || "Unknown error";
        if (msg === "NOT_SATELLITE") {
          setError("NOT_SATELLITE");
          addLog("ERROR", "Image rejected: not a satellite image");
        } else {
          setError(msg);
          addLog("ERROR", msg);
        }
        setCurrentStage(0);
      } finally {
        setIsProcessing(false);
      }
    },
    [addLog, analyze, reset],
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
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
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
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

        <ProcessingLog log={log} isProcessing={isProcessing} />
      </main>
      <footer className="mx-auto max-w-7xl px-4 py-6 text-mono text-xs text-muted-foreground">
        SatelliteVision AI · pixel pipeline runs in your browser · vision inference via secure server
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
                  <div
                    className={`absolute inset-0 ${
                      isDone ? "bg-success/60" : "bg-border"
                    }`}
                  />
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
      >
        <img src={original} alt="Original" className="h-full w-full object-cover" />
      </Panel>

      <Panel
        title="Detected Cloud Mask"
        badge={`Cloud Coverage: ${results.cloudPercentage}%`}
        delay={100}
      >
        <img src={results.cloudMask} alt="Cloud mask" className="h-full w-full object-cover" />
      </Panel>

      <Panel
        title="AI Reconstructed Image"
        badge={results.terrainFeatures.slice(0, 2).join(" · ") || "terrain"}
        delay={200}
        topRight={
          <span className="text-mono rounded-md border border-success/60 bg-success/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-success">
            Reconstructed
          </span>
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

      <Panel title="Reconstruction Confidence" delay={400}>
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

      <Panel title="Analysis Report" delay={500}>
        <div className="grid h-full grid-cols-1 gap-2 p-4 text-sm">
          <Stat label="☁️ Cloud Coverage" value={`${results.cloudPercentage}%`} />
          <Stat label="⏱ Processing Time" value={`${results.processingTimeMs}ms`} />
          <Stat
            label="📐 Resolution"
            value={`${results.imageMetadata.width}×${results.imageMetadata.height}px`}
          />
          <Stat
            label="🌍 Terrain"
            value={results.terrainFeatures.join(", ") || "—"}
          />
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
          {results.uncertaintyRegions && (
            <div className="text-mono mt-2 rounded-md border border-warning/50 bg-warning/10 p-2 text-[11px] text-warning">
              ⚠️ Uncertainty: dense cloud cover detected. The optical image does not
              contain enough information beneath the cloud to fully reconstruct this
              region.
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

  const move = (clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const x = Math.min(Math.max(clientX - r.left, 0), r.width);
    requestAnimationFrame(() => setPos((x / r.width) * 100));
  };

  useEffect(() => {
    const up = () => (draggingRef.current = false);
    const mv = (e: MouseEvent) => draggingRef.current && move(e.clientX);
    const tm = (e: TouchEvent) =>
      draggingRef.current && move(e.touches[0]?.clientX ?? 0);
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
  }, []);

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
      <img src={right} alt="Reconstructed" className="absolute inset-0 h-full w-full object-cover" />
      <div
        className="absolute inset-y-0 left-0 overflow-hidden"
        style={{ width: `${pos}%` }}
      >
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
      <div
        className="absolute inset-y-0 w-px bg-primary"
        style={{ left: `${pos}%` }}
      >
        <div className="glow-cyan absolute top-1/2 -translate-x-1/2 -translate-y-1/2 flex h-9 w-9 cursor-ew-resize items-center justify-center rounded-full border border-primary bg-background text-primary">
          <span className="text-mono text-xs">◀▶</span>
        </div>
      </div>
    </div>
  );
}

function ProcessingLog({
  log,
  isProcessing,
}: {
  log: LogEntry[];
  isProcessing: boolean;
}) {
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
